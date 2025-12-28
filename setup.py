from setuptools import setup, find_packages

setup(
    name='fucking_arx_mujoco',                          # 包名
    version='0.1.1',                       # 版本号
    packages=find_packages(),             # 自动发现包
    install_requires=[                    # 包依赖
        # 'numpy', 'torch', 'opencv-python',  # 示例依赖项
    ],
    include_package_data=True,             # 包含其他文件
    package_data={                         # 额外文件
    }
)