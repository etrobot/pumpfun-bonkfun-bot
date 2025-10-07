#!/usr/bin/env python3
"""
集成的模拟交易dashboard服务
启动后自动开始模拟交易，通过浏览器实时查看交易数据
"""

import asyncio
import json
import threading
import time
import random
import webbrowser
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Any
import signal
import sys

# 确保能导入项目模块
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.trading.dry_run import DryRunTrader, PriceActionStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TradingSimulator:
    """模拟交易引擎"""
    
    def __init__(self):
        self.strategy = PriceActionStrategy()
        self.trader = DryRunTrader(
            initial_balance=1.0,  # 1 SOL初始资金
            trade_amount=0.1,     # 每次交易0.1 SOL
            price_action_strategy=self.strategy
        )
        
        # 模拟代币池
        self.mock_tokens = [
            {"mint": "PEPE1111111111111111111111111111", "symbol": "PEPE", "name": "Pepe Coin"},
            {"mint": "DOGE1111111111111111111111111111", "symbol": "DOGE", "name": "Doge Token"},
            {"mint": "CHAD1111111111111111111111111111", "symbol": "CHAD", "name": "Chad Coin"},
            {"mint": "MOON1111111111111111111111111111", "symbol": "MOON", "name": "Moon Token"},
            {"mint": "WOJK1111111111111111111111111111", "symbol": "WOJAK", "name": "Wojak"},
        ]
        
        self.running = False
        self.portfolio_history = []
        self.recent_trades = []
        
    async def start_simulation(self):
        """开始模拟交易"""
        self.running = True
        logger.info("🤖 模拟交易开始...")
        
        while self.running:
            try:
                # 随机选择代币
                token_info = random.choice(self.mock_tokens)
                mint = token_info["mint"]
                
                # 模拟价格波动
                base_price = random.uniform(0.0000001, 0.000001)
                volatility = random.uniform(0.8, 1.5)
                
                # 生成价格序列
                prices = []
                trend = random.choice([-1, 1])  # 趋势方向
                for i in range(20):
                    noise = random.uniform(-0.1, 0.1)
                    trend_factor = trend * 0.03 * i
                    price = base_price * volatility * (1 + trend_factor + noise)
                    price = max(price, 0.0000001)
                    prices.append(price)
                    
                    self.strategy.add_price_data(mint, price)
                    await asyncio.sleep(0.2)
                
                # 决定交易动作
                current_price = prices[-1]
                
                # 检查是否已有持仓
                has_position = mint in self.trader.positions and self.trader.positions[mint].is_open
                
                if not has_position and self.trader.current_balance >= self.trader.trade_amount:
                    # 买入条件：上升趋势 + 随机概率
                    if trend > 0 and random.random() < 0.4:
                        success = await self.simulate_trade(token_info, "buy", current_price)
                        if success:
                            self.log_trade(token_info, "buy", current_price)
                
                elif has_position:
                    position = self.trader.positions[mint]
                    pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                    
                    # 卖出条件：止盈/止损/随机退出
                    should_sell = (
                        pnl_pct >= 20 or      # 止盈20%
                        pnl_pct <= -15 or     # 止损15%
                        random.random() < 0.2  # 20%随机退出
                    )
                    
                    if should_sell:
                        success = await self.simulate_trade(token_info, "sell", current_price)
                        if success:
                            self.log_trade(token_info, "sell", current_price, pnl_pct)
                
                # 更新投资组合快照
                self.update_portfolio_snapshot()
                
                # 随机等待（增加间隔以降低CPU使用）
                await asyncio.sleep(random.uniform(8, 15))
                
            except Exception as e:
                logger.exception("模拟交易错误")
                await asyncio.sleep(5)
    
    async def simulate_trade(self, token_info, action, price):
        """模拟执行交易"""
        try:
            from src.interfaces.core import TokenInfo, Platform
            from solders.pubkey import Pubkey
            
            token = TokenInfo(
                platform=Platform.PUMP_FUN,
                mint=Pubkey.from_string("11111111111111111111111111111112"),  # 占位符
                symbol=token_info["symbol"],
                name=token_info["name"],
                uri="https://example.com/metadata.json"  # 添加必需的uri参数
            )
            
            return await self.trader.execute(token, action)
        except Exception as e:
            logger.error(f"交易执行失败: {e}")
            return False
    
    def log_trade(self, token_info, action, price, pnl_pct=None):
        """记录交易"""
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "symbol": token_info["symbol"],
            "mint": token_info["mint"],
            "price": price,
            "amount": self.trader.trade_amount,
            "balance": self.trader.current_balance
        }
        
        if pnl_pct is not None:
            trade_record["pnl_pct"] = pnl_pct
        
        self.recent_trades.append(trade_record)
        if len(self.recent_trades) > 30:  # 减少交易记录数量
            self.recent_trades = self.recent_trades[-30:]
        
        print(f"📈 {action.upper()} {token_info['symbol']} @ {price:.8f} SOL" + 
              (f" (P&L: {pnl_pct:+.2f}%)" if pnl_pct else ""))
    
    def update_portfolio_snapshot(self):
        """更新投资组合快照"""
        portfolio_value = self.trader.current_balance
        for position in self.trader.positions.values():
            if position.is_open:
                portfolio_value += position.current_value(position.current_price)
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "value": portfolio_value,
            "balance": self.trader.current_balance,
            "positions": len([p for p in self.trader.positions.values() if p.is_open])
        }
        
        self.portfolio_history.append(snapshot)
        if len(self.portfolio_history) > 50:  # 减少历史记录数量
            self.portfolio_history = self.portfolio_history[-50:]
    
    def get_dashboard_data(self):
        """获取dashboard数据"""
        # 计算统计数据
        total_trades = len(self.recent_trades)
        winning_trades = len([t for t in self.recent_trades if t.get("pnl_pct", 0) > 0])
        losing_trades = len([t for t in self.recent_trades if t.get("pnl_pct", 0) < 0])
        
        current_portfolio = self.portfolio_history[-1]["value"] if self.portfolio_history else 1.0
        initial_balance = 1.0
        total_return = ((current_portfolio - initial_balance) / initial_balance) * 100
        
        return {
            "summary": {
                "portfolio_value": current_portfolio,
                "initial_balance": initial_balance,
                "total_return_pct": total_return,
                "current_balance": self.trader.current_balance,
                "total_trades": total_trades
            },
            "stats": {
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": (winning_trades / max(total_trades, 1)) * 100,
                "active_positions": len([p for p in self.trader.positions.values() if p.is_open])
            },
            "portfolio_history": self.portfolio_history[-20:],  # 最近20个数据点
            "recent_trades": self.recent_trades[-15:][::-1]     # 最近15笔交易，倒序
        }
    
    def stop(self):
        """停止模拟交易"""
        self.running = False


class DashboardHandler(SimpleHTTPRequestHandler):
    """Dashboard HTTP处理器"""
    
    def __init__(self, *args, simulator=None, **kwargs):
        self.simulator = simulator
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.serve_dashboard()
        elif self.path == '/api/data':
            self.serve_data()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        """提供dashboard页面"""
        try:
            # 读取独立的HTML文件
            dashboard_path = Path(__file__).parent / "dashboard.html"
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                html = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        except FileNotFoundError:
            raise FileNotFoundError("Dashboard HTML文件未找到")
    
    def serve_data(self):
        """提供数据API"""
        data = self.simulator.get_dashboard_data()
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
 
class TradingDashboardServer:
    """集成的交易dashboard服务器"""
    
    def __init__(self, port=8080, host="localhost"):
        self.port = port
        self.host = host
        self.simulator = TradingSimulator()
        self.server = None
        self.server_thread = None
        self.simulation_task = None
        
    def create_handler(self):
        """创建带有simulator的处理器"""
        simulator = self.simulator
        
        class HandlerWithSimulator(DashboardHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, simulator=simulator, **kwargs)
        
        return HandlerWithSimulator
    
    def start_server(self):
        """启动HTTP服务器"""
        try:
            handler_class = self.create_handler()
            self.server = HTTPServer((self.host, self.port), handler_class)
            
            self.server_thread = threading.Thread(
                target=self.server.serve_forever, 
                daemon=True
            )
            self.server_thread.start()
            
            url = f"http://{self.host}:{self.port}"
            print(f"🌐 Dashboard服务器启动: {url}")
            
            # 尝试自动打开浏览器
            try:
                webbrowser.open(url)
                print("🔗 浏览器已自动打开")
            except:
                print("💡 请手动打开浏览器访问上述地址")
            
            return True
            
        except Exception as e:
            print(f"❌ 服务器启动失败: {e}")
            return False
    
    async def start_simulation(self):
        """启动模拟交易"""
        print("🤖 启动模拟交易引擎...")
        self.simulation_task = asyncio.create_task(self.simulator.start_simulation())
        return self.simulation_task
    
    def stop(self):
        """停止服务"""
        print("\n🛑 正在停止服务...")
        
        if self.simulator:
            self.simulator.stop()
        
        if self.simulation_task:
            self.simulation_task.cancel()
        
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        print("✅ 服务已停止")


async def main():
    """主函数"""
    print("🚀 启动集成模拟交易Dashboard服务")
    print("=" * 50)
    
    # 创建服务器
    server = TradingDashboardServer(port=8080)
    
    # 设置信号处理
    def signal_handler(signum, frame):
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动HTTP服务器
    if not server.start_server():
        return
    
    print("📊 Dashboard服务器运行中...")
    print("🔄 正在启动模拟交易...")
    print("💡 访问 http://localhost:8080 查看实时数据")
    print("⏹️  按 Ctrl+C 停止服务")
    
    try:
        # 启动模拟交易并保持运行
        simulation_task = asyncio.create_task(server.simulator.start_simulation())
        
        # 保持主程序运行
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 用户主动停止服务...")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"❌ 模拟交易错误: {e}")
    finally:
        server.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 用户主动退出")
    except Exception as e:
        print(f"❌ 程序错误: {e}")