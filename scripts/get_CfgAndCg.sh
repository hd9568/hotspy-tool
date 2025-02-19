#!/bin/bash
rm -rf ./cfg_pictures
find . -maxdepth 1 -name "*.ll" -delete
find . -maxdepth 1 -name "*.dot" -delete
# 创建存储cfg图片的目录
mkdir -p ./cfg_pictures

if [[ "$1" == *[Cc]* ]]; then
    front="clang"
else
    front="flang-new"
fi


for f_file in *.$1; do
    # 移除文件扩展名，获取基本文件名
    base_name=$(basename "$f_file" .$1)

    # 将.f文件转换为.ll文件
    $front -S -emit-llvm -I./ -I../ "$f_file" -o "${base_name}.ll"

    # 将.ll文件转换为.dot文件
    opt -passes=dot-cfg "${base_name}.ll"

    # 遍历生成的.dot文件并转换为png图片
done

#for dot_file in *.dot; do
#    dot -Tpng "$dot_file" -o "./cfg_pictures/${base_name}_$(basename "$dot_file" .dot).png"
#done

# find . -maxdepth 1 -name "*.dot" | while read dot_file; do
#     # 提取不带路径的基本文件名
#     base_name=$(basename "$dot_file" .dot)

#     # 执行 dot 命令
#     dot -Tpng "$dot_file" -o "./cfg_pictures/${base_name}.png"
# done


rm -rf ./cfg_dot
mkdir -p ./cfg_dot

# 查找并移动所有.dot文件到cfg_dot目录
find . -maxdepth 1 -name "*.dot" -exec mv {} ./cfg_dot/ \;

sleep 4

## get callgraph

rm -rf ./cg_pictures
find . -maxdepth 1 -name "*.ll" -delete
find . -maxdepth 1 -name "*.dot" -delete
# 创建存储cfg图片的目录
mkdir -p ./cg_pictures


for f_file in *.$1; do
    # 移除文件扩展名，获取基本文件名
    base_name=$(basename "$f_file" .$1)

    # 将.f文件转换为.ll文件
    $front -S -emit-llvm -I./ -I../ "$f_file" -o "${base_name}.ll"

    # 将.ll文件转换为.dot文件
    opt -passes=dot-callgraph "${base_name}.ll"

    # 遍历生成的.dot文件并转换为png图片
done
sleep 2

#for dot_file in *.dot; do
#    dot -Tpng "$dot_file" -o "./cfg_pictures/${base_name}_$(basename "$dot_file" .dot).png"
#done

# find . -maxdepth 1 -name "*.dot" | while read dot_file; do
#     # 提取不带路径的基本文件名
#     base_name=$(basename "$dot_file" .dot)

#     # 执行 dot 命令
#     dot -Tpng "$dot_file" -o "./cg_pictures/${base_name}.png"
# done

sleep 2


rm -rf ./cg_dot
mkdir -p ./cg_dot

# 查找并移动所有.dot文件到cfg_dot目录
find . -maxdepth 1 -name "*.dot" -exec mv {} ./cg_dot/ \;

