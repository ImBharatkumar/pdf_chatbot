import sys 
from src.logger import logging
import traceback

import traceback

def error_message_detail(error_message, error_detail=None):
    if error_detail is not None:
        error_detail_msg = f'Error Detail:\n{error_detail}'
    else:
        error_detail_msg = ''

    if type(error_message) is str:
        error_message = error_message.replace('\n', '<br>')

    return f'<pre>{error_message}{error_detail_msg}</pre>'
    

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    

   