from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime


class ProviderMsg_MessageContent(BaseModel):
    """Base content that all message types must provide"""

    type: str
    raw_content: Dict[str, Any] = Field(
        description="Original provider-specific content"
    )


class ProviderMsg_TextContent(ProviderMsg_MessageContent):
    type: str = "text"
    text: str


class ProviderMsg_ImageContent(ProviderMsg_MessageContent):
    type: str = "image"
    url: Optional[str] = None
    content_provider: Dict[str, Any]
    preview_url: Optional[str] = None


class ProviderMsg_VideoContent(ProviderMsg_MessageContent):
    type: str = "video"
    url: Optional[str] = None
    content_provider: Dict[str, Any]
    duration: Optional[int] = None
    preview_url: Optional[str] = None


class ProviderMsg_ProviderMsg_AudioContent(ProviderMsg_MessageContent):
    type: str = "audio"
    url: Optional[str] = None
    content_provider: Dict[str, Any]
    duration: int


class ProviderMsg_LocationContent(ProviderMsg_MessageContent):
    type: str = "location"
    title: Optional[str] = None
    address: Optional[str] = None
    latitude: float
    longitude: float


class ProviderMsg_StickerContent(ProviderMsg_MessageContent):
    type: str = "sticker"
    package_id: str
    sticker_id: str
    keywords: Optional[List[str]] = None


class ProviderMsg_FileContent(ProviderMsg_MessageContent):
    type: str = "file"
    filename: str
    file_size: int
    file_type: Optional[str] = None


class ProviderMessage(BaseModel):
    """Standardized message format for all providers"""

    provider: str = Field(description="Message provider (e.g., 'line', 'discord')")
    message_id: str = Field(description="Provider's message ID")
    user_id: str = Field(description="User ID in provider's system")
    reply_token: Optional[str] = None
    timestamp: datetime
    content: Union[
        ProviderMsg_TextContent,
        ProviderMsg_ImageContent,
        ProviderMsg_VideoContent,
        ProviderMsg_ProviderMsg_AudioContent,
        ProviderMsg_LocationContent,
        ProviderMsg_StickerContent,
        ProviderMsg_FileContent,
    ]
    thread_id: Optional[str] = None
    reply_to: Optional[str] = None
    mentions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
