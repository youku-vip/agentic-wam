#!/usr/bin/env python3
"""Read-only test of the robot SDK's Euler/rotation6D conventions; sends no robot commands."""
import json
import numpy as np
from scipy.spatial.transform import Rotation


def check_conventions(trans_utils):
    # A mixed-axis pose distinguishes row/column conventions; identity alone cannot.
    angles = np.array([[0.31, -0.27, 0.44], [-0.41, 0.19, -0.33], [0.2, 0.4, -0.1]])
    encoded = np.asarray([trans_utils.euler_angle_to_rot6d(pose) for pose in angles], dtype=float)
    if encoded.shape != (3, 6) or not np.isfinite(encoded).all():
        raise ValueError('SDK returned invalid rotation6D')
    candidates = []
    for order in ('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx', 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'):
        for degrees in (False, True):
            matrices = Rotation.from_euler(order, angles, degrees=degrees).as_matrix()
            for encoding in ('first_two_columns', 'first_two_rows'):
                expected = (np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=-1)
                            if encoding == 'first_two_columns' else matrices[:, :2, :].reshape(3, 6))
                candidates.append(dict(euler_order=order, angle_unit='degree' if degrees else 'radian',
                                       encoding=encoding, max_error=float(np.max(np.abs(encoded - expected)))))
    best = min(candidates, key=lambda candidate: candidate['max_error'])
    decoded = [trans_utils.rot6d_to_euler_angle(value) for value in encoded]
    roundtrip = np.asarray([trans_utils.euler_angle_to_rot6d(value) for value in decoded])
    error = float(np.max(np.abs(roundtrip - encoded)))
    return dict(best_match=best, sdk_pair_roundtrip_max_error=error,
                first_two_columns_verified=best['encoding'] == 'first_two_columns' and best['max_error'] < 1e-5,
                sdk_pair_verified=error < 1e-5, robot_commands_sent=0)


def main():
    from trans_tools import trans_utils
    result = check_conventions(trans_utils)
    print(json.dumps(result, indent=2))
    if not result['first_two_columns_verified'] or not result['sdk_pair_verified']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
