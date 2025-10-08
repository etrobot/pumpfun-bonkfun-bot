"""
代理配置工具模块
"""

import os
from typing import Optional, Dict, Any
import aiohttp
from utils.logger import get_logger

logger = get_logger(__name__)


class ProxyConfig:
    """代理配置管理类"""
    
    def __init__(self):
        self.use_proxy = os.getenv("USE_PROXY", "false").lower() == "true"
        self.http_proxy = os.getenv("HTTP_PROXY")
        self.https_proxy = os.getenv("HTTPS_PROXY")
        
        if self.use_proxy:
            logger.info(f"代理已启用 - HTTP: {self.http_proxy}, HTTPS: {self.https_proxy}")
        else:
            logger.info("代理未启用")
    
    def get_proxy_url(self, target_url: str) -> Optional[str]:
        """根据目标URL获取相应的代理URL"""
        if not self.use_proxy:
            return None
            
        if target_url.startswith('https://'):
            return self.https_proxy
        elif target_url.startswith('http://'):
            return self.http_proxy
        else:
            return self.http_proxy  # 默认使用HTTP代理
    
    def get_aiohttp_session_kwargs(self) -> Dict[str, Any]:
        """获取aiohttp会话的配置参数"""
        kwargs = {}

        if self.use_proxy:
            # For HTTPS through HTTP proxy, we need both trust_env and a connector
            # that doesn't interfere with proxy CONNECT tunneling
            kwargs['trust_env'] = True
            # Don't verify SSL when using proxy to avoid certificate issues
            kwargs['connector'] = aiohttp.TCPConnector(ssl=False)

        return kwargs
    
    def get_request_kwargs(self, target_url: str) -> Dict[str, Any]:
        """获取请求的配置参数"""
        kwargs = {}
        
        proxy_url = self.get_proxy_url(target_url)
        if proxy_url:
            kwargs['proxy'] = proxy_url
            
        return kwargs
    
    def set_env_proxy(self):
        """设置环境变量代理（用于不直接支持代理的库）"""
        if self.use_proxy:
            if self.http_proxy:
                os.environ['HTTP_PROXY'] = self.http_proxy
            if self.https_proxy:
                os.environ['HTTPS_PROXY'] = self.https_proxy
    
    def restore_env_proxy(self, original_http: Optional[str] = None, original_https: Optional[str] = None):
        """恢复原始环境变量"""
        if original_http is not None:
            os.environ['HTTP_PROXY'] = original_http
        elif 'HTTP_PROXY' in os.environ:
            del os.environ['HTTP_PROXY']
            
        if original_https is not None:
            os.environ['HTTPS_PROXY'] = original_https
        elif 'HTTPS_PROXY' in os.environ:
            del os.environ['HTTPS_PROXY']


# 全局代理配置实例
proxy_config = ProxyConfig()