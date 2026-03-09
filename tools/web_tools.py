#!/usr/bin/env python3
"""
网页工具模块
Web Tools Module

提供维基百科搜索、网页搜索、网页读取等功能
"""

import os
import re
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime


class WikipediaSearch:
    """
    维基百科搜索
    提供无限知识库扩展
    """
    
    def __init__(self, cache_size: int = 100, cache_file: str = None):
        self.cache: Dict[str, Dict] = {}
        self.cache_size = cache_size
        self.enabled = True
        self.cache_file = cache_file
        
        # 加载缓存
        if cache_file and os.path.exists(cache_file):
            self._load_cache()
    
    def _load_cache(self):
        """加载缓存"""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.cache = data.get("cache", {})
        except:
            pass
    
    def _save_cache(self):
        """保存缓存"""
        if not self.cache_file:
            return
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump({"cache": self.cache}, f, ensure_ascii=False)
        except:
            pass
    
    def search(self, query: str, sentences: int = 3) -> Optional[str]:
        """
        搜索维基百科
        
        Args:
            query: 搜索查询
            sentences: 返回的句子数量
            
        Returns:
            搜索结果摘要
        """
        if not self.enabled:
            return None
        
        # 检查缓存
        cache_key = query.lower()
        if cache_key in self.cache:
            return self.cache[cache_key].get("content")
        
        try:
            import wikipedia
            
            # 设置语言
            if self._is_chinese(query):
                wikipedia.set_lang("zh")
            else:
                wikipedia.set_lang("en")
            
            # 搜索
            results = wikipedia.search(query, results=5)
            
            if results:
                try:
                    page = wikipedia.page(results[0], auto_suggest=False)
                    summary = wikipedia.summary(results[0], sentences=sentences)
                    
                    # 缓存结果
                    self._add_to_cache(cache_key, {
                        "content": summary,
                        "title": page.title,
                        "url": page.url
                    })
                    
                    return summary
                    
                except wikipedia.DisambiguationError as e:
                    # 尝试第一个选项
                    try:
                        summary = wikipedia.summary(e.options[0], sentences=sentences)
                        return summary
                    except:
                        pass
                        
                except wikipedia.PageError:
                    pass
                        
        except ImportError:
            print("提示: 安装wikipedia模块以启用维基百科搜索: pip install wikipedia")
            
        except Exception as e:
            pass
        
        return None
    
    def _is_chinese(self, text: str) -> bool:
        """检测是否为中文"""
        return any('\u4e00' <= c <= '\u9fff' for c in text)
    
    def _add_to_cache(self, key: str, value: Dict):
        """添加到缓存"""
        if len(self.cache) >= self.cache_size:
            # 删除最旧的条目
            oldest = min(self.cache.items(), key=lambda x: x[1].get("timestamp", 0))
            del self.cache[oldest[0]]
        
        value["timestamp"] = time.time()
        self.cache[key] = value
        self._save_cache()


class WebSearch:
    """
    网页搜索
    使用DuckDuckGo API
    """
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.enabled = True
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        网页搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if not self.enabled:
            return []
        
        try:
            import requests
            
            # 使用DuckDuckGo API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            data = response.json()
            
            results = []
            
            # 主要结果
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", ""),
                    "snippet": data.get("AbstractText", ""),
                    "url": data.get("AbstractURL", ""),
                    "source": "duckduckgo"
                })
            
            # 相关主题
            for topic in data.get("RelatedTopics", [])[:num_results-1]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("FirstURL", "").split("/")[-1] if topic.get("FirstURL") else "",
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "source": "duckduckgo"
                    })
            
            return results
            
        except ImportError:
            print("提示: 安装requests模块以启用网页搜索: pip install requests")
            return []
            
        except Exception as e:
            return [{"error": str(e)}]
    
    def read_page(self, url: str, max_length: int = 5000) -> Optional[str]:
        """
        读取网页内容
        
        Args:
            url: 网页URL
            max_length: 最大内容长度
            
        Returns:
            网页文本内容
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=self.timeout)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 移除脚本和样式
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 提取文本
            text = soup.get_text(separator=' ', strip=True)
            
            # 清理多余空白
            text = re.sub(r'\s+', ' ', text)
            
            return text[:max_length]
            
        except ImportError:
            print("提示: 安装beautifulsoup4以启用网页读取: pip install beautifulsoup4")
            return None
            
        except Exception as e:
            return None


class ToolManager:
    """
    工具管理器
    统一管理所有工具
    """
    
    def __init__(self):
        self.wiki = WikipediaSearch()
        self.web = WebSearch()
        
        self.tools = {
            "wiki_search": self._wiki_search,
            "web_search": self._web_search,
            "read_page": self._read_page
        }
    
    def call(self, tool_name: str, **kwargs) -> Dict:
        """
        调用工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        if tool_name not in self.tools:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        
        try:
            result = self.tools[tool_name](**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _wiki_search(self, query: str, sentences: int = 3) -> Optional[str]:
        """维基百科搜索"""
        return self.wiki.search(query, sentences)
    
    def _web_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """网页搜索"""
        return self.web.search(query, num_results)
    
    def _read_page(self, url: str, max_length: int = 5000) -> Optional[str]:
        """读取网页"""
        return self.web.read_page(url, max_length)
    
    def list_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self.tools.keys())


def main():
    """测试工具"""
    print("=" * 60)
    print("网页工具测试")
    print("=" * 60)
    
    manager = ToolManager()
    
    # 测试维基百科搜索
    print("\n[1] 维基百科搜索测试")
    result = manager.call("wiki_search", query="人工智能", sentences=2)
    print(f"结果: {result}")
    
    # 测试网页搜索
    print("\n[2] 网页搜索测试")
    result = manager.call("web_search", query="Python programming", num_results=3)
    print(f"结果: {result}")
    
    print("\n" + "=" * 60)
    print("测试完成")


if __name__ == "__main__":
    main()
