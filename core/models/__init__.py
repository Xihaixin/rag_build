"""
core.models — 数据模型定义

本模块定义了所有业务流使用的数据模型（dataclass），
对应前端 page.tsx 和 Ask.tsx 中的 TypeScript 接口。

数据模型:
  - WikiPage: Wiki 页面（对应前端 WikiPage 接口）
  - WikiSection: Wiki 章节（对应前端 WikiSection 接口）
  - WikiStructure: Wiki 结构（对应前端 WikiStructure 接口）
  - Message: 聊天消息（对应前端 Message 接口）
  - ResearchStage: 研究阶段（对应前端 ResearchStage 接口）
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WikiPage:
    """Wiki 页面 — 对应前端 page.tsx 中的 WikiPage 接口"""
    id: str
    title: str
    content: str = ""
    filePaths: List[str] = field(default_factory=list)
    importance: str = "medium"  # 'high' | 'medium' | 'low'
    relatedPages: List[str] = field(default_factory=list)
    parentId: Optional[str] = None
    isSection: bool = False
    children: List[str] = field(default_factory=list)


@dataclass
class WikiSection:
    """Wiki 章节 — 对应前端 page.tsx 中的 WikiSection 接口"""
    id: str
    title: str
    pages: List[str] = field(default_factory=list)
    subsections: List[str] = field(default_factory=list)


@dataclass
class WikiStructure:
    """Wiki 结构 — 对应前端 page.tsx 中的 WikiStructure 接口"""
    id: str
    title: str
    description: str = ""
    pages: List[WikiPage] = field(default_factory=list)
    sections: List[WikiSection] = field(default_factory=list)
    rootSections: List[str] = field(default_factory=list)


@dataclass
class Message:
    """聊天消息 — 对应前端 Ask.tsx 中的 Message 接口"""
    role: str  # 'user' | 'assistant' | 'system'
    content: str


@dataclass
class ResearchStage:
    """研究阶段 — 对应前端 Ask.tsx 中的 ResearchStage 接口"""
    title: str
    content: str
    iteration: int
    type: str  # 'plan' | 'update' | 'conclusion'


__all__ = [
    "WikiPage",
    "WikiSection",
    "WikiStructure",
    "Message",
    "ResearchStage",
]
