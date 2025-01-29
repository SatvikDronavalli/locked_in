from IMU_Data import X_train, X_test
from Force_Data import GRF_right_train, GRF_left_train, GRF_right_test, GRF_left_test
import numpy as np
X_train_combined = np.stack([GRF_right_train, GRF_left_train], axis=-1)
X_test_combined = np.stack([GRF_right_test, GRF_left_test], axis=-1)
