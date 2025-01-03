from typing import List, Optional

algorithm: str = None
metric: str = None
graph: Optional[bool] = False
block:Optional[tuple] = (32, 32)
video: Optional[str] = 'data/video.mp4'
frame: Optional[int] = 0
secondframe: Optional[int] = None
level: Optional[int] = 3
block_size: Optional[int] = 16
search_window: Optional[int] = 16
score_algorithm: Optional[str] = "MI"