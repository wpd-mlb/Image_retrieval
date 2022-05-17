'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/1/27 23:07
'''
import pynvml

def get_free_device_ids():
    pynvml.nvmlInit()
    num_device = pynvml.nvmlDeviceGetCount()
    free_device_id = []
    for i in range(num_device):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        men_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("GPU", i, ":", pynvml.nvmlDeviceGetName(handle))
        # print(men_info.total/1024/1024/1024,men_info.free/1024/1024/1024) # 原始单位是B(Byte)字节
        # GPU 0 : b'GeForce GTX 1060 with Max-Q Design'--6GB
        # import pdb; pdb.set_trace()
        if men_info.free >= men_info.total*0.8:
            free_device_id.append(i)
    return free_device_id


if __name__ == "__main__":
    pass
    # 简单使用
    # https: // blog.csdn.net / u014636245 / article / details / 83932090
    # https://docs.python.org/zh-cn/3/library/pdb.html
    # https://zhuanlan.zhihu.com/p/37294138

    print(get_free_device_ids())
    # import pdb; pdb.set_trace()