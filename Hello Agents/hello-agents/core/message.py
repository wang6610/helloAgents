# 消息系统
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel
from datetime import datetime

MessageRole = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    role : MessageRole
    content : str
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None   # 元数据，存放额外信息

    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get('timestamp', datetime.now()),
            metadata=kwargs.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（OpenAI API格式）"""
        return {
            "role": self.role,
            "content": self.content
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"