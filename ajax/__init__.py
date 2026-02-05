"""
AJAX OS Overlay - Acceso directo
Uso:
    import ajax
    bot = ajax.wake_up()
    bot.do("tarea")
"""

from agency.ajax_core import AjaxCore, wake_up, wake_up_chat_lite

__all__ = ["AjaxCore", "wake_up", "wake_up_chat_lite"]
