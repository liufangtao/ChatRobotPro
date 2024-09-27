from PIL import Image
from io import BytesIO  
import numpy as np

def rgb_array_to_jpeg_bytes(rgb_array):  
    """  
    将NumPy RGB数组转换为JPEG字节流。  
      
    参数:  
        rgb_array (numpy.ndarray): 一个形状为(height, width, 3)的NumPy数组，表示RGB图像。  
          
    返回:  
        bytes: JPEG格式的图像字节流。  
    """  
    # 确保输入是NumPy数组，并且有3个通道（RGB）  
    if not isinstance(rgb_array, np.ndarray) or rgb_array.shape[2] != 3:  
        raise ValueError("Input must be a NumPy array with shape (height, width, 3).")  
      
    # 将NumPy数组转换为PIL图像  
    image = Image.fromarray(rgb_array.astype(np.uint8), 'RGB')  
      
    # 创建一个BytesIO流对象，并将PIL图像保存为JPEG格式到这个流中  
    jpeg_bytes = BytesIO()  
    image.save(jpeg_bytes, format='JPEG')  
      
    # 获取JPEG字节流的值，并将流的位置重置到开始位置以便读取  
    jpeg_data = jpeg_bytes.getvalue()  
    jpeg_bytes.close()  
      
    return jpeg_data
