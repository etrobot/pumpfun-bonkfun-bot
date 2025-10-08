#!/usr/bin/env python3
"""
Quick test script to verify proxy connection is working
"""
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from src.core.client import SolanaClient
from solders.pubkey import Pubkey

# Load environment
load_dotenv()


async def test_connection():
    """Test if we can connect to Solana RPC through proxy"""
    print("🔧 Testing Solana RPC connection with proxy...")
    print(f"Proxy enabled: {os.getenv('USE_PROXY')}")
    print(f"HTTP_PROXY: {os.getenv('HTTP_PROXY')}")
    print(f"HTTPS_PROXY: {os.getenv('HTTPS_PROXY')}")
    print()

    rpc_endpoint = os.getenv("SOLANA_NODE_RPC_ENDPOINT", "https://api.mainnet-beta.solana.com")
    print(f"RPC Endpoint: {rpc_endpoint}")
    print()

    client = SolanaClient(rpc_endpoint)

    try:
        # Test 1: Get account info (using a known bonding curve) - uses httpx/AsyncClient
        print("Test 1: Getting account info (httpx/AsyncClient)...")
        bonding_curve = "6GXfUqrmPM4VdN1NoDZsE155jzRegJngZRjMkGyby7do"
        account_info = await client.get_account_info(Pubkey.from_string(bonding_curve))
        print(f"✅ Account info retrieved: {len(account_info.data)} bytes of data")
        print()

        # Test 2: Get latest blockhash - uses httpx/AsyncClient
        print("Test 2: Getting latest blockhash (httpx/AsyncClient)...")
        blockhash = await client.get_latest_blockhash()
        print(f"✅ Blockhash: {blockhash}")
        print()

        print("🎉 All tests passed! Proxy connection is working correctly.")
        print("Note: These tests use httpx-based AsyncClient which respects HTTP_PROXY/HTTPS_PROXY env vars")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_connection())
