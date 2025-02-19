import os
import json
import subprocess
import ctypes

class FunctionData:
    def __init__(self,program="default_program", name="default_name", cfg="default_cfg", cgs="default_cgs", label=0):
        self.program = program
        self.name = name  # 函数名
        self.cfg = cfg    # 控制流图（Control Flow Graph）
        self.cgs = cgs      # 调用图（Call Graph）
        self.label = label  # 标签，可以用于分类或其他目的

    def __str__(self):
        return f"Function Name: {self.name}\n cfg_dir:{self.cfg}\n cgs:{self.cgs}\n Label: {self.label}\n"
    
    def to_dict(self):
        return {
            "program":self.program,
            "name":self.name,
            "cfg":self.cfg,
            "cgs":self.cgs,
            "label":self.label
        }


def demangle(command):
    command = "c++filt "+command
    try:
        # 运行 shell 命令
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # 检查命令是否成功执行
        if result.returncode == 0:
            # 返回输出结果
            return result.stdout.strip()
        else:
            # 返回错误信息
            return result.stderr.strip()
    except Exception as e:
        return str(e)


def find_string_in_file(file_path, target_string):
    # 打开文件进行读取
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            strs  = line.split("\"{")
            if(len(strs)>1):
                strs = strs[1].split("}\"")
                if target_string == strs[0]:
                    return True
    return False


# 获取当前目录
current_directory = '/home/hd/gp/trainSet'
json_name = "train.json"
functions = []
total0 = 0
total1 = 0

# 遍历当前目录下的所有条目
for entry in os.listdir(current_directory):
    # 拼接完整的文件/文件夹路径
    full_path = os.path.join(current_directory, entry)
    # 检查这个条目是否是一个文件夹
    if os.path.isdir(full_path):
        # 初始化一个字典来存储名称和对应的数值
        name_value_pairs = {}
        # 打开文件
        with open(full_path+"/"+entry+".txt", 'r',encoding="utf-8") as file:
            # 逐行读取文件
            for line in file:
                # 分割每一行
                parts = line.split()
                # 获取名称（行中的第一个元素）
                name = ' '.join(parts[:-1])  # 这假设名称可能由多个单词组成
                # print(name)
                # 获取数值（行中的最后一个元素）
                value = parts[-1]
                # 将名称和数值存储在字典中
                name_value_pairs[name] = value
        for entry1 in os.listdir(full_path+"/cfg_dot"):
            function = FunctionData()
            function.program = entry
            function.cgs = []
            cfg_dir = full_path+"/cfg_dot/"+entry1
            function.cfg = cfg_dir
            function.name = entry1[1:].split(".")[0]
            name = ""
            cg_exist = False
            name = demangle(function.name)
            if(name in name_value_pairs):
                function.label = name_value_pairs[name]
            for entry2 in os.listdir(full_path+"/cg_dot"):
                found = find_string_in_file(full_path+"/cg_dot"+"/"+entry2, function.name)
                if found:
                    cg_exist = True
                    function.cgs.append(full_path+"/cg_dot"+"/"+entry2)
                    break
            if(name in name_value_pairs and cg_exist):
                functions.append(function)
                if(len(function.cgs)==0):
                    print(function.cfg)
                    print("error! no cg")
                    exit()
                if(function.label == '0'):
                    total0 = total0 + 1
                else:
                    total1 = total1 + 1
                  
print(len(functions))
print("sum of 0 is "+str(total0)+" ; "+"sum of 1 is "+str(total1))
functions_dict_list = (function.to_dict() for function in functions)

with open(json_name, 'w') as file:
    json.dump(list(functions_dict_list), file, ensure_ascii=False, indent=4)


        