# testmytranspose.py

import unittest
import numpy as np
import pandas as pd

# my_transpose_solution.py 파일에서 mytranspose 함수를 불러옴
from my_transpose_solution import mytranspose

# torch 라이브러리가 있을 경우에만 사용하도록 설정
try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

# unittest.TestCase를 상속받는 테스트 클래스 생성
class TestMyTranspose(unittest.TestCase):

    # (1) Matrix의 경우
    def test_numpy_matrix(self):
        # 5x2 행렬 테스트
        myvar1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        transposed = mytranspose(myvar1)
        self.assertEqual(transposed.shape, (2, 5))
        np.testing.assert_array_equal(transposed, myvar1.T)

        # 1x2 행렬 테스트
        myvar2 = np.array([[1, 2]])
        transposed2 = mytranspose(myvar2)
        self.assertEqual(transposed2.shape, (2, 1))

        # 2x1 행렬 테스트
        myvar3 = np.array([[1], [2]])
        transposed3 = mytranspose(myvar3)
        self.assertEqual(transposed3.shape, (1, 2))
        
        # 빈 행렬 테스트
        myvar4 = np.empty((0, 0))
        transposed4 = mytranspose(myvar4)
        self.assertEqual(transposed4.shape, (0, 0))

    # (2) Vector의 경우
    def test_numpy_vector(self):
        # 기본 벡터 테스트
        myvar1 = np.array([1, 2, 3, 4])
        transposed = mytranspose(myvar1)
        self.assertEqual(transposed.shape, (4, 1))

        # NaN 포함 벡터 테스트
        myvar2 = np.array([1, 2, np.nan, 3])
        transposed2 = mytranspose(myvar2)
        # np.testing.assert_array_equal은 NaN도 같은 위치에 있으면 같다고 처리해줌
        np.testing.assert_array_equal(transposed2, np.array([[1], [2], [np.nan], [3]]))
        
        # 빈 배열 테스트
        myvar3 = np.array([])
        transposed3 = mytranspose(myvar3)
        self.assertEqual(transposed3.shape, (0, 1))

    # (3) DataFrame의 경우
    def test_pandas_dataframe(self):
        D = np.array([1, 2, 3, 4])
        E = np.array(["red ", " white ", " red ", np.nan])
        F = np.array([True, True, True, False])
        df = pd.DataFrame({"d": D, "e": E, "f": F})
        
        transposed_df = mytranspose(df)
        
        # pandas에서 제공하는 테스트 함수로 비교
        pd.testing.assert_frame_equal(transposed_df, df.T)

    # (4) PyTorch Tensor의 경우
    # torch가 설치되어 있지 않으면 이 테스트는 건너뜀
    @unittest.skipIf(not _torch_available, "PyTorch 라이브러리가 설치되어 있지 않습니다.")
    def test_pytorch_tensor(self):
        np_array = np.array([[1, 2], [3, 4]])
        tensor_pt = torch.tensor(np_array)
        
        transposed_tensor = mytranspose(tensor_pt)
        
        # torch.equal 함수로 두 텐서가 같은지 비교
        self.assertTrue(torch.equal(transposed_tensor, tensor_pt.T))

# 이 파일을 직접 실행했을 때 unittest를 실행하도록 하는 코드
if __name__ == '__main__':
    unittest.main()