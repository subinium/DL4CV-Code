# for easy checking
from pytorch101 import *

# 1
hello()

# 2
print(create_sample_tensor())

# 3
indices = [(0, 0), (1, 1), (2, 2)]
values = [1, 2, 3]
x = torch.zeros((3, 3))
print(mutate_tensor(x, indices, values))

# 4
print(count_tensor_elements(torch.zeros((2, 3, 4))))

# 5
print(create_tensor_of_pi(3, 4))

# 6
print(multiples_of_ten(5, 25))

# 7
x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 8, 10], [11, 12, 13, 14, 15]])
print(slice_indexing_practice(x))

# 8
x = torch.zeros(5, 7, dtype=torch.int64)
print(slice_assignment_practice(x))

# 9
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(shuffle_cols(x))

# 10
print(reverse_rows(x))

# 11
print(take_one_elem_per_col(x))

# 12
x = torch.tensor([[-1, -1, 0], [0, 1, 2], [3, 4, 5]])
print(count_negative_entries(x))

# 13
x = [1, 4, 3, 2]
print(make_one_hot(x))

# 14
x = torch.arange(24)
print(reshape_practice(x))

# 15
x = torch.tensor([
    [10, 20, 30],
    [2,  5,  1]
])
print(zero_row_min(x))

# 16
B, N, M, P = 2, 3, 5, 4
x = torch.randn(B, N, M)
y = torch.randn(B, M, P)
print(batched_matrix_multiply(x, y, True))
print(batched_matrix_multiply(x, y, False))

# 17
x = torch.tensor([[0., 30., 600.], [1., 10., 200.], [-1., 20., 400.]])
print(normalize_columns(x))
