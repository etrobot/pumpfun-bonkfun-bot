#!/usr/bin/env python3
"""
集成的模拟交易dashboard服务
启动后自动开始模拟交易，通过浏览器实时查看交易数据
"""

import asyncio
import json
import os
import signal
import sys
import threading
import time
import random
import webbrowser
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Any
import base64
import struct

import base58
import websockets

# 确保能导入项目模块
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ 已加载.env配置文件")
except ImportError:
    print("⚠️ python-dotenv未安装，使用系统环境变量")

from src.trading.dry_run import DryRunTrader, PriceActionStrategy
from src.core.client import SolanaClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TradingSimulator:
    """真实行情交易引擎"""
    
    def __init__(self):
        # 初始化Solana客户端
        rpc_endpoint = os.getenv("SOLANA_NODE_RPC_ENDPOINT")
        if not rpc_endpoint:
            logger.warning("未设置 SOLANA_NODE_RPC_ENDPOINT，使用公共RPC端点（可能受限）")
            rpc_endpoint = "https://api.mainnet-beta.solana.com"

        logger.info(f"使用RPC端点: {rpc_endpoint}")
        print(f"🔗 连接RPC端点: {rpc_endpoint}")

        # 测试连接
        print("🔍 正在测试RPC连接...")
        try:
            self.solana_client = SolanaClient(rpc_endpoint)
            print("✅ Solana客户端初始化成功")
        except Exception as e:
            print(f"❌ Solana客户端初始化失败: {e}")
            logger.error(f"Solana客户端初始化失败: {e}")
            raise
        
        self.strategy = PriceActionStrategy()
        self.trader = DryRunTrader(
            initial_balance=1.0,  # 1 SOL初始资金
            trade_amount=0.1,     # 每次交易0.1 SOL
            price_action_strategy=self.strategy
        )
        
        # 真实pump.fun代币配置（支持环境变量覆盖）
        env_symbol = os.getenv("TARGET_TOKEN_SYMBOL")
        env_name = os.getenv("TARGET_TOKEN_NAME") or env_symbol
        env_mint = os.getenv("TARGET_TOKEN_MINT")
        env_curve = os.getenv("TARGET_BONDING_CURVE")

        if env_symbol and env_curve:
            # 使用环境变量作为唯一目标代币
            self.real_tokens = [
                {
                    "mint": env_mint or "",
                    "symbol": env_symbol,
                    "name": env_name or env_symbol,
                    "bonding_curve": env_curve,
                }
            ]
            logger.info(
                f"使用环境变量配置目标代币: {env_symbol} (curve={env_curve[:6]}... )"
            )
        else:
            # 默认代币列表（占位，便于演示）。如有需要请通过环境变量覆盖。
            self.real_tokens = [
                {
                    "mint": "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr",
                    "symbol": "POPCAT",
                    "name": "Popcat",
                    "bonding_curve": "6GXfUqrmPM4VdN1NoDZsE155jzRegJngZRjMkGyby7do",
                }
            ]
            logger.info("未提供环境变量，使用默认代币 POPCAT")
        
        self.running = False
        self.portfolio_history = []
        self.recent_trades = []
        self.activity_logs = []  # 新增：活动日志队列
        # 当前目标代币（由嗅探器或外部设置）
        self.current_token = self.real_tokens[0] if self.real_tokens else None

        logger.info("交易模拟器已初始化，使用真实pump.fun代币价格")
    
    def set_target_token(self, token_info: dict):
        """设置当前目标代币（由嗅探器调用）"""
        self.current_token = token_info
        symbol = token_info.get("symbol", "?")
        self.add_activity_log(f"嗅探到新代币，切换目标: {symbol}", log_type="success")
        logger.info(f"切换当前目标代币为: {symbol}")
    
    async def get_real_price(self, bonding_curve_address: str) -> float:
        """获取真实的bonding curve价格"""
        try:
            import struct
            from construct import Flag, Int64ul, Struct
            from solders.pubkey import Pubkey

            # 常量定义
            LAMPORTS_PER_SOL = 1_000_000_000
            TOKEN_DECIMALS = 6
            EXPECTED_DISCRIMINATOR = struct.pack("<Q", 6966180631402821399)

            # Bonding curve状态结构
            BondingCurveState = Struct(
                "virtual_token_reserves" / Int64ul,
                "virtual_sol_reserves" / Int64ul,
                "real_token_reserves" / Int64ul,
                "real_sol_reserves" / Int64ul,
                "token_total_supply" / Int64ul,
                "complete" / Flag,
            )

            # 获取账户信息（添加超时控制）
            logger.debug(f"正在获取bonding curve账户信息: {bonding_curve_address}")
            curve_address = Pubkey.from_string(bonding_curve_address)

            # 使用超时控制避免长时间等待
            account_info = await asyncio.wait_for(
                self.solana_client.get_account_info(curve_address),
                timeout=10.0  # 10秒超时
            )

            if not account_info or not hasattr(account_info, 'data'):
                logger.warning(f"No data in bonding curve account {bonding_curve_address}")
                return 0.0

            # 解码数据 (solders返回bytes类型)
            data = account_info.data

            if data[:8] != EXPECTED_DISCRIMINATOR:
                logger.warning(f"Invalid curve state discriminator for {bonding_curve_address}")
                return 0.0

            # 解析bonding curve状态
            curve_state = BondingCurveState.parse(data[8:])
            
            if curve_state.virtual_token_reserves <= 0 or curve_state.virtual_sol_reserves <= 0:
                logger.warning(f"Invalid reserve state for {bonding_curve_address}")
                return 0.0
            
            # 计算价格: SOL储备 / 代币储备
            price = (curve_state.virtual_sol_reserves / LAMPORTS_PER_SOL) / (
                curve_state.virtual_token_reserves / 10**TOKEN_DECIMALS
            )

            return price

        except asyncio.TimeoutError:
            logger.error(f"获取真实价格超时 {bonding_curve_address}")
            return 0.0
        except Exception as e:
            logger.exception(f"获取真实价格失败 {bonding_curve_address}: {e}")
            return 0.0

    async def _get_fallback_price(self, token_info: dict) -> float:
        """降级获取模拟价格"""
        import random
        base_price = 0.000001  # 基础价格
        price_multiplier = random.uniform(0.5, 5.0)
        volatility = 1 + (random.random() - 0.5) * 0.2
        simulated_price = base_price * price_multiplier * volatility
        return max(simulated_price, 0.0000001)

    def add_activity_log(self, message: str, log_type: str = "info"):
        """添加活动日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "type": log_type  # info, success, warning, error
        }
        self.activity_logs.append(log_entry)
        # 保持最近100条日志
        if len(self.activity_logs) > 100:
            self.activity_logs = self.activity_logs[-100:]

    async def start_simulation(self):
        """开始真实行情交易"""
        self.running = True
        logger.info("🤖 真实行情交易开始...")
        print("💡 模拟交易引擎已启动，开始获取实时价格...")

        while self.running:
            try:
                # 选择目标代币（由嗅探器或默认设置）
                token_info = self.current_token
                if not token_info:
                    self.add_activity_log("暂无目标代币，等待嗅探...", log_type="info")
                    await asyncio.sleep(3)
                    continue
                mint = token_info["mint"]
                bonding_curve = token_info["bonding_curve"]

                print(f"\n🔍 正在分析代币: {token_info['symbol']}")
                self.add_activity_log(f"正在分析代币: {token_info['symbol']}")
                logger.info(f"开始获取 {token_info['symbol']} 的价格数据")

                # 获取真实价格数据
                current_price = await self.get_real_price(bonding_curve)

                # 如果无法获取真实价格，使用模拟价格
                if current_price <= 0:
                    logger.warning(f"无法获取 {token_info['symbol']} 的真实价格，使用模拟价格")
                    print(f"⚠️  {token_info['symbol']} 价格获取失败，使用模拟数据")
                    self.add_activity_log(
                        f"{token_info['symbol']} 价格获取失败，使用模拟数据", log_type="warning"
                    )
                    current_price = await self._get_fallback_price(token_info)

                if current_price <= 0:
                    logger.warning(f"代币 {token_info['symbol']} 价格异常: {current_price}")
                    self.add_activity_log(
                        f"代币 {token_info['symbol']} 价格异常: {current_price}", log_type="warning"
                    )
                    await asyncio.sleep(10)
                    continue

                logger.info(f"📊 获取 {token_info['symbol']} 实时价格: {current_price:.10f} SOL")
                print(f"💰 {token_info['symbol']} 当前价格: {current_price:.10f} SOL")
                self.add_activity_log(
                    f"{token_info['symbol']} 当前价格: {current_price:.10f} SOL"
                )
                
                # 收集一段时间的真实价格数据
                prices = []
                print(f"📈 开始收集价格数据...")
                self.add_activity_log("开始收集价格数据...")
                for i in range(5):  # 收集5个价格点
                    try:
                        price = await self.get_real_price(bonding_curve)
                        if price <= 0:
                            # 降级使用模拟价格
                            price = await self._get_fallback_price(token_info)

                        if price > 0:
                            prices.append(price)
                            self.strategy.add_price_data(mint, price)

                            # 定期更新投资组合快照
                            if i % 2 == 0:
                                self.update_portfolio_snapshot()

                            logger.debug(f"{token_info['symbol']} 价格 #{i+1}: {price:.10f} SOL")
                            print(f"  ├─ 数据点 {i+1}/5: {price:.10f} SOL")
                            self.add_activity_log(
                                f"数据点 {i+1}/5: {price:.10f} SOL"
                            )
                        else:
                            logger.warning(f"代币 {token_info['symbol']} 价格为零，跳过")
                            self.add_activity_log(
                                f"代币 {token_info['symbol']} 价格为零，跳过", log_type="warning"
                            )

                    except Exception as e:
                        logger.error(f"获取价格数据失败: {e}")
                        self.add_activity_log(
                            f"获取价格数据失败: {e}", log_type="error"
                        )

                    await asyncio.sleep(3)  # 每3秒获取一次价格
                
                if not prices:
                    logger.warning(f"未能获取 {token_info['symbol']} 的有效价格数据")
                    self.add_activity_log(
                        f"未能获取 {token_info['symbol']} 的有效价格数据", log_type="warning"
                    )
                    await asyncio.sleep(10)
                    continue
                
                # 分析价格趋势
                current_price = prices[-1]
                if len(prices) >= 2:
                    price_change = (current_price - prices[0]) / prices[0]
                    trend = 1 if price_change > 0 else -1
                else:
                    trend = random.choice([-1, 1])
                
                # 检查是否已有持仓
                has_position = mint in self.trader.positions and self.trader.positions[mint].status == "open"
                
                if not has_position and self.trader.current_balance >= self.trader.trade_amount:
                    # 买入条件：上升趋势 + 更高的随机概率
                    if trend > 0 and random.random() < 0.8:  # 提高到80%概率
                        success = await self.simulate_trade(token_info, "buy", current_price)
                        if success:
                            self.log_trade(token_info, "buy", current_price)
                
                elif has_position:
                    position = self.trader.positions[mint]
                    pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                    
                    # 卖出条件：止盈/止损/随机退出
                    should_sell = (
                        pnl_pct >= 15 or      # 降低止盈到15%
                        pnl_pct <= -10 or     # 降低止损到10%
                        random.random() < 0.4  # 提高到40%随机退出
                    )
                    
                    if should_sell:
                        success = await self.simulate_trade(token_info, "sell", current_price)
                        if success:
                            self.log_trade(token_info, "sell", current_price, pnl_pct)
                
                # 更新投资组合快照
                self.update_portfolio_snapshot()
                
                # 等待下一次价格检查（真实行情检查间隔）
                await asyncio.sleep(random.uniform(30, 60))  # 30-60秒检查一次，避免过于频繁的API调用
                
            except Exception as e:
                logger.exception("模拟交易错误")
                await asyncio.sleep(5)
    
    async def simulate_trade(self, token_info, action, _price):
        """模拟执行交易"""
        try:
            from src.interfaces.core import TokenInfo, Platform
            from solders.pubkey import Pubkey
            
            # 使用实际的mint地址而不是占位符
            mint_str = token_info["mint"]
            
            token = TokenInfo(
                platform=Platform.PUMP_FUN,
                mint=Pubkey.from_string(mint_str),
                symbol=token_info["symbol"],
                name=token_info["name"],
                uri="https://example.com/metadata.json"
            )
            
            # 直接调用内部方法，绕过价格分析限制
            if action == "buy":
                result = await self.trader._simulate_buy(token, time.time())
            elif action == "sell":
                result = await self.trader._simulate_sell(token, time.time())
            else:
                return False
                
            return result.success
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
        # 同步到活动日志
        if action == "buy":
            self.add_activity_log(
                f"买入 {token_info['symbol']} @ {price:.8f} SOL", log_type="success"
            )
        elif action == "sell":
            if pnl_pct is not None:
                log_type = "success" if pnl_pct >= 0 else "warning"
                self.add_activity_log(
                    f"卖出 {token_info['symbol']} @ {price:.8f} SOL (P&L: {pnl_pct:+.2f}%)",
                    log_type=log_type,
                )
            else:
                self.add_activity_log(
                    f"卖出 {token_info['symbol']} @ {price:.8f} SOL",
                    log_type="success",
                )
    
    def update_portfolio_snapshot(self):
        """更新投资组合快照"""
        # 计算当前投资组合总价值
        portfolio_value = self.trader.current_balance
        
        # 为每个开放持仓计算当前价值
        for mint, position in self.trader.positions.items():
            if position.status == "open":
                # 获取该代币的最新价格
                latest_price = self.strategy.get_latest_price(mint)
                if latest_price:
                    # 计算持仓当前价值 = 持仓数量 * 当前价格
                    token_amount = position.amount_sol / position.entry_price  # 代币数量
                    current_value = token_amount * latest_price
                    portfolio_value += current_value
                else:
                    # 如果没有最新价格，使用成本价
                    portfolio_value += position.amount_sol
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "value": portfolio_value,
            "balance": self.trader.current_balance,
            "positions": len([p for p in self.trader.positions.values() if p.status == "open"])
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
                "active_positions": len([p for p in self.trader.positions.values() if p.status == "open"])
            },
            "portfolio_history": self.portfolio_history[-20:],  # 最近20个数据点
            "recent_trades": self.recent_trades[-15:][::-1],    # 最近15笔交易，倒序
            "activity_logs": self.activity_logs[-50:]           # 最近活动日志
        }
    
    def stop(self):
        """停止模拟交易"""
        self.running = False


class DashboardHandler(SimpleHTTPRequestHandler):
    """Dashboard HTTP处理器"""

    def __init__(self, *args, **kwargs):
        # 从kwargs中提取simulator，避免传递给父类
        self.simulator = kwargs.pop('simulator', None)
        try:
            super().__init__(*args, **kwargs)
        except (ConnectionResetError, BrokenPipeError) as e:
            # 客户端提前关闭连接，静默处理
            logger.debug(f"客户端连接已关闭: {e}")
        except Exception as e:
            # 记录其他未预期的错误
            logger.error(f"请求处理错误: {e}")
    
    def log_message(self, format, *args):
        """重写日志方法，只记录重要请求"""
        # 只记录非SSE和非favicon的请求
        if not (self.path.startswith('/api/stream') or self.path == '/favicon.ico'):
            super().log_message(format, *args)
    
    def do_GET(self):
        if self.path == '/':
            self.serve_dashboard()
        elif self.path == '/api/data':
            self.serve_data()
        elif self.path == '/api/stream':
            self.serve_sse_stream()
        elif self.path == '/favicon.ico':
            # 返回空的favicon，避免404错误
            self.send_response(204)  # No Content
            self.end_headers()
        else:
            # 静默处理未知请求，减少日志噪音
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not Found')
    
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
    
    def serve_sse_stream(self):
        """提供SSE数据流"""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Cache-Control')
        self.end_headers()
        
        try:
            # 发送初始数据
            data = self.simulator.get_dashboard_data()
            self.send_sse_data(data)
            
            # 持续发送更新数据
            while True:
                time.sleep(3)  # 每3秒更新一次
                try:
                    data = self.simulator.get_dashboard_data()
                    self.send_sse_data(data)
                except BrokenPipeError:
                    # 客户端断开连接，正常退出
                    logger.debug("SSE客户端断开连接")
                    break
                except Exception as e:
                    logger.error(f"SSE数据发送失败: {e}")
                    break
                    
        except BrokenPipeError:
            logger.debug("SSE连接被客户端关闭")
        except Exception as e:
            logger.error(f"SSE连接错误: {e}")
    
    def send_sse_data(self, data):
        """发送SSE格式的数据"""
        try:
            json_data = json.dumps(data, ensure_ascii=False)
            sse_message = f"data: {json_data}\n\n"
            self.wfile.write(sse_message.encode('utf-8'))
            self.wfile.flush()
        except BrokenPipeError:
            # 客户端断开连接，正常情况
            logger.debug("客户端断开SSE连接")
            raise
        except Exception as e:
            logger.error(f"SSE消息发送失败: {e}")
            raise
 
class TradingDashboardServer:
    """集成的交易dashboard服务器"""
    
    def __init__(self, port=8080, host="localhost"):
        self.port = port
        self.host = host
        self.simulator = TradingSimulator()
        self.server = None
        self.server_thread = None
        self.simulation_task = None
        self.sniffer_task = None
        
    def create_handler(self):
        """创建带有simulator的处理器"""
        simulator = self.simulator

        class HandlerWithSimulator(DashboardHandler):
            def __init__(self, *args, **kwargs):
                # 直接设置simulator，不通过kwargs传递
                kwargs['simulator'] = simulator
                try:
                    super().__init__(*args, **kwargs)
                except (ConnectionResetError, BrokenPipeError):
                    # 客户端提前关闭连接，静默处理
                    pass
                except Exception:
                    # 其他错误也静默处理，避免服务器崩溃
                    pass

        return HandlerWithSimulator
    
    async def start_sniffer(self):
        """基于LogSubscribe的pump.fun新币嗅探器"""
        # 仅跟踪 pump.fun Program
        pump_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
        # 优先使用项目规范的变量名，其次兼容示例里的变量名
        wss_endpoint = (
            os.getenv("SOLANA_RPC_WEBSOCKET")
            or os.getenv("SOLANA_NODE_WSS_ENDPOINT")
        )
        if not wss_endpoint:
            self.simulator.add_activity_log(
                "未配置 SOLANA_RPC_WEBSOCKET，嗅探器未启动", log_type="warning"
            )
            return
        self.simulator.add_activity_log(
            "嗅探器已启动（LogSubscribe，仅跟踪 pump.fun）", log_type="info"
        )

        async def parse_create_instruction(data_bytes: bytes):
            # 参照 learning-examples 的解析
            try:
                if len(data_bytes) < 8:
                    return None
                offset = 8
                parsed = {}
                fields = [
                    ("name", "string"),
                    ("symbol", "string"),
                    ("uri", "string"),
                    ("mint", "publicKey"),
                    ("bondingCurve", "publicKey"),
                    ("user", "publicKey"),
                    ("creator", "publicKey"),
                ]
                for fname, ftype in fields:
                    if ftype == "string":
                        length = struct.unpack("<I", data_bytes[offset: offset+4])[0]
                        offset += 4
                        value = data_bytes[offset: offset+length].decode("utf-8")
                        offset += length
                    else:
                        value = base58.b58encode(data_bytes[offset: offset+32]).decode("utf-8")
                        offset += 32
                    parsed[fname] = value
                return parsed
            except Exception:
                return None

        while True:
            try:
                async with websockets.connect(wss_endpoint) as ws:
                    sub_msg = json.dumps({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "logsSubscribe",
                        "params": [
                            {"mentions": [pump_program_id]},
                            {"commitment": "processed"},
                        ],
                    })
                    await ws.send(sub_msg)
                    _ = await ws.recv()  # subscription confirmation
                    self.simulator.add_activity_log("已订阅 pump.fun 日志", log_type="info")

                    while True:
                        try:
                            resp = await ws.recv()
                            payload = json.loads(resp)
                            if payload.get("method") != "logsNotification":
                                continue
                            result = payload["params"]["result"]["value"]
                            logs = result.get("logs", [])

                            if any("Program log: Instruction: Create" in l for l in logs):
                                # 找 Program data
                                for l in logs:
                                    if "Program data:" in l:
                                        try:
                                            encoded = l.split(": ")[1]
                                            decoded = base64.b64decode(encoded)
                                            parsed = await parse_create_instruction(decoded)
                                            if parsed and parsed.get("symbol") and parsed.get("bondingCurve"):
                                                token_info = {
                                                    "symbol": parsed.get("symbol"),
                                                    "name": parsed.get("name") or parsed.get("symbol"),
                                                    "mint": parsed.get("mint", ""),
                                                    "bonding_curve": parsed.get("bondingCurve"),
                                                }
                                                self.simulator.set_target_token(token_info)
                                        except Exception as e:
                                            self.simulator.add_activity_log(
                                                f"嗅探解析失败: {e}", log_type="warning"
                                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            self.simulator.add_activity_log(
                                f"嗅探接收错误: {e}", log_type="warning"
                            )
                            break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.simulator.add_activity_log(
                    f"嗅探连接错误: {e}，5秒后重连...", log_type="warning"
                )
                await asyncio.sleep(5)
    
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
        if self.sniffer_task:
            self.sniffer_task.cancel()
        
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
    def signal_handler(_signum, _frame):
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
        _simulation_task = asyncio.create_task(server.simulator.start_simulation())
        # 启动嗅探器（仅跟踪 pump.fun）
        server.sniffer_task = asyncio.create_task(server.start_sniffer())

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