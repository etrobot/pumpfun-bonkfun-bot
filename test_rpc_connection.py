#!/usr/bin/env python3
"""测试RPC连接"""

import asyncio
import os
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from src.core.client import SolanaClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_connection():
    """测试RPC连接"""
    rpc_endpoint = os.getenv("SOLANA_NODE_RPC_ENDPOINT", "https://api.mainnet-beta.solana.com")
    print(f"🔗 测试RPC端点: {rpc_endpoint}")
    print(f"🌐 代理设置: HTTP_PROXY={os.getenv('HTTP_PROXY')}, HTTPS_PROXY={os.getenv('HTTPS_PROXY')}")
    print(f"🎛️  USE_PROXY={os.getenv('USE_PROXY')}")

    try:
        print("\n1️⃣ 正在初始化Solana客户端...")
        client = SolanaClient(rpc_endpoint)
        print("✅ 客户端初始化成功")

        print("\n2️⃣ 等待blockhash缓存初始化...")
        await asyncio.sleep(2)

        print("\n3️⃣ 测试健康检查...")
        health = await asyncio.wait_for(client.get_health(), timeout=10)
        print(f"✅ 健康状态: {health}")

        print("\n4️⃣ 测试获取blockhash...")
        blockhash = await asyncio.wait_for(client.get_latest_blockhash(), timeout=10)
        print(f"✅ Blockhash: {blockhash}")

        print("\n✅ 所有测试通过!")

        await client.close()
        return True

    except asyncio.TimeoutError as e:
        print(f"\n❌ 超时错误: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 连接失败: {e}")
        logger.exception("连接测试失败")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)
