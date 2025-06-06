# my_transpose_solution.py

import numpy as np
import pandas as pd

# torch 라이브러리가 있을 경우에만 사용하도록 설정
try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

def mytranspose(data):
    """
    다양한 데이터 타입(NumPy 배열, Pandas DataFrame, PyTorch Tensor)을
    입력받아 전치(transpose)하여 반환합니다.
    """

    # (1) Pandas DataFrame의 경우
    if isinstance(data, pd.DataFrame):
        # DataFrame은 .T 라는 편리한 기능이 있어서 사용
        return data.T

    # (2) PyTorch Tensor의 경우
    elif _torch_available and isinstance(data, torch.Tensor):
        # 1차원 벡터이면 (n,) -> (n, 1) 형태로 변환
        if data.ndim == 1:
            return data.unsqueeze(1)
        # 2차원 이상이면 .T 기능 사용
        else:
            return data.T

    # (3) NumPy Array의 경우 
    elif isinstance(data, np.ndarray):
        # 1차원 벡터이면 (n,) -> (n, 1) 형태로 변환
        if data.ndim == 1:
            # reshape을 이용해 모양 변경
            return data.reshape(-1, 1)

        # 2차원 행렬이면 직접 전치 구현 
        else:
            num_rows = data.shape[0]  # 원래 행의 수
            num_cols = data.shape[1]  # 원래 열의 수

            # 전치된 행렬은 행/열 수가 반대이므로, 그 크기에 맞게 0으로 채운 새 행렬 생성
            transposed_array = np.zeros((num_cols, num_rows))

            # 반복문을 돌면서 값 하나하나 옮기기
            for i in range(num_rows):      # 원래 행을 기준으로 반복
                for j in range(num_cols):  # 원래 열을 기준으로 반복
                    # 새 행렬의 (j, i) 위치에 원래 행렬의 (i, j) 값을 넣음
                    transposed_array[j, i] = data[i, j]

            return transposed_array

    # 지원하지 않는 데이터 타입의 경우
    else:
        return None # 혹은 에러를 발생시킬 수도 있음