# Convert Libero dataset to AInnoRobotDatasets

## About Libero dataset

It is from https://github.com/Lifelong-Robot-Learning/LIBERO
We employed BAKU https://github.com/siddhanthaldar/BAKU/ to generate a intermediate dataset.
See https://github.com/siddhanthaldar/BAKU/blob/main/baku/data_generation/generate_libero.py

After that, we run below command to convert the intermediate dataset to AInnoRobotDatasets format.
```bash
python convert_libero_to_ainno.py
```

Some notes: If set a breakpoint at ainno_robot_data_schema/converters/convert_libero/libero.py:_sample function and inspect below vars, we see:
```python
(Pdb++) episode['observation']['robot_states'].shape 
(124, 9)
(Pdb++) episode['observation']['joint_states'].shape  
(124, 7)
(Pdb++) episode['action'].shape
(124, 7)
(Pdb++)  episode['observation']['gripper_states'].shape
(124, 2)
(Pdb++) episode['observation']['robot_states'][0]
array([ 3.6208291e-02, -3.6248561e-02, -1.9008571e-01, -3.6267458e-05,
        1.1628076e+00,  9.9924082e-01,  2.3110198e-02, -3.0655034e-02,
        6.6280076e-03], dtype=float32)
(Pdb++)  episode['observation']['gripper_states'][0]
array([ 0.03403868, -0.03407298], dtype=float32)
(Pdb++) episode['action'][0]
array([ 0.45267856, -0.05892857, -0.        , -0.01285714,  0.11142857,
        0.04392857, -1.        ], dtype=float32)
(Pdb++) episode['observation']['joint_states'][0] 
array([ 4.4694198e-03, -1.0663578e-01,  2.4581456e-04, -2.4116230e+00,
       -1.6058763e-02,  2.2517982e+00,  7.5959408e-01], dtype=float32)
(Pdb++) episode['observation']['robot_states'][5]
array([ 0.03882367, -0.03887209, -0.16414627, -0.00415987,  1.158391  ,
        0.99757695,  0.03782748, -0.05740864,  0.01065531], dtype=float32)
(Pdb++) episode['observation'].keys()
dict_keys(['robot_states', 'pixels', 'pixels_egocentric', 'joint_states', 'eef_states', 'gripper_states'])
(Pdb++) episode['observation']['gripper_states'][5]
array([ 0.03870665, -0.03874977], dtype=float32)
(Pdb++) episode['observation']['joint_states'][5]
array([ 2.1870967e-03, -6.0873185e-03, -2.0019454e-03, -2.2962751e+00,
       -2.8526649e-02,  2.1873903e+00,  7.3420733e-01], dtype=float32)
(Pdb++) episode['observation']['eef_states'][5]
array([-0.1695318 , -0.00319221,  1.1591046 ,  0.99801296,  0.03472643,
       -0.05164202,  0.00986488], dtype=float32)
(Pdb++) episode['action'][5].shape
(7,)
(Pdb++) episode['action'].shape
(124, 7)
```

So we infer that:
- episode['observation']['robot_states'] is 9-elements, and it is combination of episode['observation']['gripper_states'] and episode['observation']['eef_states']
- episode['observation']['gripper_states'] is 2-elements, and the 2 values are just opposite values, e.g. 0.03870665, -0.03874977