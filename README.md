A Python package that provides a secondary wrapper for the official Python API of the optical simulation software Lumerical, enabling easier usage in more complex scenarios.

Features:

1. Wrap the original lsf (Lumerical script language) and make it pythonic

   `addpower;set("name","power monitor");set("x",0);`→`addpower(name="power monitor",x=0)`

2. Provides utility functions such as active instance management `get_fdtd_instance()` and GPU acceleration `GPU_on()`.

3. Provide a utility function library for design of Diffractive Optical Neural Network`donn.py`

The Lumerpy package has been released on PyPI, so you can easily install it using the following command: `pip install lumerpy`

To use this package, you must have Lumerical installed and obtain the Lumerical GUI License authorization. The default location of the lumapi (officially provided API) is: `C:\Program Files\Lumerical\v241\api\python`.

For a quick start, you can copy all the contents of `consol.py` and run them in the Python console.

Official Lumerical Python API overview: [Lumerical](https://optics.ansys.com/hc/en-us/articles/360037824513-Python-API-overview)

Github: [lumerpy](https://github.com/oscarxchen/lumerpy)

License: [MIT](https://github.com/OscarXChen/lumerpy/blob/main/LICENSE.txt)

More detailed information will be updated later. Current version of the README file: 1.3

------

一个基于光学仿真软件Lumerical官方Python API二次包装的Python库，以便于在更复杂情况下使用。

主要内容包括：

1. 将Lumerical脚本语言风格二次包装为Python语言风格

   `addpower;set("name","power monitor");set("x",0);`→`addpower(name="power monitor",x=0)`

2. 提供了如活动实例管理 `get_fdtd_instance()` GPU加速 `GPU_on()` 等实用函数

3. 提供了用于衍射光学神经网络设计的实用函数库`donn.py`

Lumerpy已发布至pypi，故可通过运行命令`pip install lumerpy`便捷安装

使用Lumerpy前，必须先正确安装Lumerical并且获得Lumerical GUI License授权。默认的lumapi（官方提供的api）地址为：`C:\Program Files\Lumerical\v241\api\python`

一个简单的示例是，你可以直接复制`consol.py`中的所有内容，并在python console中运行

Lumerical 官方 Python API 概览: [Lumerical](https://optics.ansys.com/hc/en-us/articles/360037824513-Python-API-overview)

Github 地址: [lumerpy](https://github.com/oscarxchen/lumerpy)

License: [MIT](https://github.com/OscarXChen/lumerpy/blob/main/LICENSE.txt)

更多的说明信息会在之后更新，README 文件当前版本：1.3

