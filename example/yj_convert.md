## TJ转换
作者：曹明浩    时间：2026.5.22
### 1、 合并一个性状所有的染色体数据
#### 1.1、 重命名文件
```shell
* rename_male_pop_paternal.list 
chra,chra
windowa,windowa
chrb,chrb
windowb,windowb
trait_id,trait_id
trait_male,trait_male
malecount,malecount
allelea,allelea
allelea_mutant_paternal,allelea_mutant_paternal
alleleb,alleleb
alleleb_mutant_paternal,alleleb_mutant_paternal
peerallelea,peerallelea
peerallelea_mutant_maternal,peerallelea_mutant_maternal
peeralleleb,peeralleleb
peeralleleb_mutant_maternal,peeralleleb_mutant_maternal
trait_male_diff_paternal,trait_value
malecount_diff_paternal,trait_count
delta_pheno,delta_pheno
delta_fraq,delta_fraq



* rename_male_pop_maternal.list
chra,chra
windowa,windowa
chrb,chrb
windowb,windowb
trait_id,trait_id
trait_male,trait_male
malecount,malecount
allelea,allelea
allelea_mutant_paternal,allelea_mutant_paternal
alleleb,alleleb
alleleb_mutant_paternal,alleleb_mutant_paternal
peerallelea,peerallelea
peerallelea_mutant_maternal,peerallelea_mutant_maternal
peeralleleb,peeralleleb
peeralleleb_mutant_maternal,peeralleleb_mutant_maternal
trait_male_diff_maternal,trait_value
malecount_diff_maternal,trait_count
delta_pheno,delta_pheno
delta_fraq,delta_fraq


* rename_female_pop_paternal.list 
chra,chra
windowa,windowa
chrb,chrb
windowb,windowb
trait_id,trait_id
trait_female,trait_female
femalecount,femalecount
allelea,allelea
allelea_mutant_paternal,allelea_mutant_paternal
alleleb,alleleb
alleleb_mutant_paternal,alleleb_mutant_paternal
peerallelea,peerallelea
peerallelea_mutant_maternal,peerallelea_mutant_maternal
peeralleleb,peeralleleb
peeralleleb_mutant_maternal,peeralleleb_mutant_maternal
trait_female_diff_paternal,trait_value
femalecount_diff_paternal,trait_count
delta_pheno,delta_pheno
delta_fraq,delta_fraq




* rename_female_pop_maternal.list
chra,chra
windowa,windowa
chrb,chrb
windowb,windowb
trait_id,trait_id
trait_female,trait_female
femalecount,femalecount
allelea,allelea
allelea_mutant_paternal,allelea_mutant_paternal
alleleb,alleleb
alleleb_mutant_paternal,alleleb_mutant_paternal
peerallelea,peerallelea
peerallelea_mutant_maternal,peerallelea_mutant_maternal
peeralleleb,peeralleleb
peeralleleb_mutant_maternal,peeralleleb_mutant_maternal
trait_female_diff_maternal,trait_value
femalecount_diff_maternal,trait_count
delta_pheno,delta_pheno
delta_fraq,delta_fraq


```
#### 1.2、 创建一个YJ转换的脚本

* 创建一个万能脚本
```shell
usage() {    
    echo "Usage: $0 -t <trait_id> -p <prefix_name> -i <input_base_path> -o <output_base_path> [-m <mode>] [-y]"
    echo "Modes:"
    echo "  male_pos_flip    - Process male positive flip data (default)"
    echo "  female_pos_flip  - Process female positive flip data"
    echo "  male_neg_flip    - Process male negative flip data"
    echo "  female_neg_flip  - Process female negative flip data"
    echo "Options:"
    echo "  -y               - Apply Yeo-Johnson transformation (default: false)"
    echo "  --no-yj          - Skip Yeo-Johnson transformation"
    echo "Example: $0 -t 112 -p poseffectmale_frequency -i /path/to/input -o /path/to/output -y"
    exit 1
}

# 初始化 trait 为空，shell中的变量是不允许有空格的
trait=""
prefix_name=""
input_base_path=""
output_base_path=""
mode="male_pos_flip"
apply_yj=false  # 默认不应用YJ转换


# 根据mode设置具体的文件命名规则
set_file_patterns() {
    local mode=$1
    case ${mode} in
        male_pos_flip)
            prefix_name="poseffectmale_frequency"
            suffix_P="_P"
            suffix_M="_M"
            rename_paternal="rename_male_pop_paternal.list"  
            rename_maternal="rename_male_pop_maternal.list"
            ;;
        female_pos_flip)
            prefix_name="poseffectfemale_frequency"
            suffix_P="_P"
            suffix_M="_M"
            rename_paternal="rename_female_pop_paternal.list"  
            rename_maternal="rename_female_pop_maternal.list"
            ;;
        male_neg_flip)
            prefix_name="negeffectmale_frequency"
            suffix_P="_P"
            suffix_M="_M"
            rename_paternal="rename_male_pop_paternal.list"  
            rename_maternal="rename_male_pop_maternal.list"
            ;;
        female_neg_flip)
            prefix_name="negeffectfemale_frequency"
            suffix_P="_P"
            suffix_M="_M"
            rename_paternal="rename_female_pop_paternal.list"  
            rename_maternal="rename_female_pop_maternal.list"
            ;;
        *)
            echo "Error: Unknown mode '$mode'"
            usage
            ;;
    esac
}

# YJ转换函数
apply_yeo_johnson() {
    local input_file=$1
    local output_file=$2
    
    # 检查输入文件是否存在
    if [[ ! -f "$input_file" ]]; then
        echo "Warning: Input file $input_file not found, skipping YJ transformation"
        return 1
    fi
    
    # 创建输出目录
    mkdir -p $(dirname "$output_file")
    
    echo "Applying Yeo-Johnson transformation to: $input_file"
    
    # 调用Python脚本进行YJ转换
    # 假设yj_trans.py在脚本同目录下，或者使用绝对路径
    local script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    local yj_script="${script_dir}/yj_trans.py"
    
    if [[ ! -f "$yj_script" ]]; then
        echo "Error: yj_trans.py not found at $yj_script"
        return 1
    fi
    
    # 执行YJ转换，转换4列：trait_value, trait_count, delta_pheno, delta_fraq
    python "$yj_script" \
        -i "$input_file" \
        -o "$output_file" \
        -c trait_value trait_count delta_pheno delta_fraq \
        -v
    
    if [[ $? -eq 0 ]]; then
        echo "Yeo-Johnson transformation completed successfully"
        return 0
    else
        echo "Error: Yeo-Johnson transformation failed"
        return 1
    fi
}

# 解析命令行参数
while getopts "t:p:i:o:m:y-:" opt; do
    case ${opt} in
        t)
            trait=$OPTARG
            ;;
        p)
            prefix_name=$OPTARG
            ;;
        i)
            input_base_path=$OPTARG
            ;;
        o)
            output_base_path=$OPTARG
            ;;
        m)
            mode=$OPTARG
            ;;
        y)
            apply_yj=true
            ;;
        -)
            case "${OPTARG}" in
                no-yj)
                    apply_yj=false
                    ;;
                *)
                    echo "Invalid option: --${OPTARG}" >&2
                    usage
                    ;;
            esac
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

# 检查必需参数
if [[ -z "$trait" ]]; then
    echo "Error: trait is required."
    usage
fi

if [[ -z "$input_base_path" ]]; then
    echo "Error: input base path is required."
    usage
fi

if [[ -z "$output_base_path" ]]; then
    echo "Error: output base path is required."
    usage
fi

# 如果提供了prefix_name，则优先使用用户指定的，否则根据mode设置
if [[ -n "$prefix_name" ]]; then
    echo "Using user-specified prefix: $prefix_name"
else
    set_file_patterns $mode
    echo "Using mode: $mode, prefix: $prefix_name"
fi

# 构建完整的输入输出路径
input_path="${input_base_path}"
output_path="${output_base_path}"

# 创建必要的目录
mkdir -p ${output_path}/01.merge_parquet
mkdir -p ${output_path}/02.new_field_merge_parquet
mkdir -p ${output_path}/03.scale/01.origin_scale
mkdir -p ${output_path}/03.scale/02.yj_scale

echo "Processing trait: $trait"
echo "Input path: $input_path"
echo "Output path: $output_path"
echo "Prefix: $prefix_name"
echo "Apply YJ transformation: $apply_yj"

# 合并parquet文件
echo "Merging P files..."
statstools concat_parquet \
    -i ${input_path}/trait${trait}/${prefix_name}_*_${trait}_P.parquet \
    -o ${output_path}/01.merge_parquet/${prefix_name}_chrom_${trait}_P.parquet

echo "Merging M files..."
statstools concat_parquet \
    -i ${input_path}/trait${trait}/${prefix_name}_*_${trait}_M.parquet \
    -o ${output_path}/01.merge_parquet/${prefix_name}_chrom_${trait}_M.parquet

# 添加恒定值列
echo "Adding constant fields..."
statstools add_constent_field \
    -i ${output_path}/01.merge_parquet/${prefix_name}_chrom_${trait}_P.parquet \
    -f origin \
    -c P

statstools add_constent_field \
    -i ${output_path}/01.merge_parquet/${prefix_name}_chrom_${trait}_M.parquet \
    -f origin \
    -c M

echo -e "\n"

# 查看原始列名
echo -e "${prefix_name}_chrom_${trait}_P.parquet 原始列名是：\n"
statstools change_field_name \
    -i ${output_path}/01.merge_parquet/${prefix_name}_chrom_${trait}_P.parquet \
    -l

echo -e "${prefix_name}_chrom_${trait}_M.parquet 原始列名是：\n"
statstools change_field_name \
    -i ${output_path}/01.merge_parquet/${prefix_name}_chrom_${trait}_M.parquet \
    -l

# 执行重命名（需要确保rename list文件存在）
echo "Renaming columns..."
if [[ -f ${output_path}/${rename_paternal} ]]; then
    statstools change_field_name \
        -i ${output_path}/01.merge_parquet/${prefix_name}_chrom_${trait}_P.parquet \
        -s ${output_path}/${rename_paternal} \
        -o ${output_path}/02.new_field_merge_parquet/${prefix_name}_chrom_${trait}_P_rename.parquet
else
    echo "Warning: ${output_path}/${rename_paternal} not found, skipping rename for P file"
    cp ${output_path}/01.merge_parquet/${prefix_name}_chrom_${trait}_P.parquet \
       ${output_path}/02.new_field_merge_parquet/${prefix_name}_chrom_${trait}_P_rename.parquet
fi

if [[ -f ${output_path}/${rename_maternal} ]]; then
    statstools change_field_name \
        -i ${output_path}/01.merge_parquet/${prefix_name}_chrom_${trait}_M.parquet \
        -s ${output_path}/${rename_maternal} \
        -o ${output_path}/02.new_field_merge_parquet/${prefix_name}_chrom_${trait}_M_rename.parquet
else
    echo "Warning: ${output_path}/${rename_maternal} not found, skipping rename for M file"
    cp ${output_path}/01.merge_parquet/${prefix_name}_chrom_${trait}_M.parquet \
       ${output_path}/02.new_field_merge_parquet/${prefix_name}_chrom_${trait}_M_rename.parquet
fi

# 查看新列名
echo -e "${prefix_name}_chrom_${trait}_P_rename.parquet 修改后列名是：\n"
statstools change_field_name \
    -i ${output_path}/02.new_field_merge_parquet/${prefix_name}_chrom_${trait}_P_rename.parquet \
    -l

echo -e "${prefix_name}_chrom_${trait}_M_rename.parquet 修改后列名是：\n"
statstools change_field_name \
    -i ${output_path}/02.new_field_merge_parquet/${prefix_name}_chrom_${trait}_M_rename.parquet \
    -l

echo -e "\n"

# 合并 M 和 P 组
echo "Merging P and M files..."
merge_output="${output_path}/03.scale/01.origin_scale/${prefix_name}_chrom_${trait}_origin.parquet"
statstools concat_parquet \
    -i ${output_path}/02.new_field_merge_parquet/${prefix_name}_chrom_${trait}_P_rename.parquet \
       ${output_path}/02.new_field_merge_parquet/${prefix_name}_chrom_${trait}_M_rename.parquet \
    -o "$merge_output"

echo "Processing completed for trait $trait"

# 应用Yeo-Johnson转换
if [[ "$apply_yj" == true ]]; then
    echo -e "\n=========================================="
    echo "Starting Yeo-Johnson transformation..."
    echo "=========================================="
    
    yj_input="$merge_output"
    yj_output="${output_path}/03.scale/02.yj_scale/${prefix_name}_chrom_${trait}_origin_yj.parquet"
    
    if apply_yeo_johnson "$yj_input" "$yj_output"; then
        echo "Yeo-Johnson transformation completed for trait $trait"
        echo "Output saved to: $yj_output"
    else
        echo "Warning: Yeo-Johnson transformation failed for trait $trait"
        exit 1
    fi
else
    echo "Skipping Yeo-Johnson transformation (use -y to enable)"
fi

echo "All processing completed for trait $trait"
```


****
==============================
****
* 提交任务的模板,male population + positive flip
```shell
#!/bin/bash
# 创建执行脚本 submit_jobs.sh
# 定义输入输出路径
INPUT_BASE="/public/home/xiehaibing7/caominghao_v11/10.male_pos_flip_parquet_data"
OUTPUT_BASE="/public/home/xiehaibing7/caominghao_v11/c12.convert_normal"

# 处理male positive flip数据
for trait in 3 9 10 14 15 16 17 18 19 25 26 59 60 63 124 139 8 48 51 52 53 54 55 67 73 75 117 121 128 129 11 21 22 23 24 31 47 50 58 62 83 84 85 86 87 88 91 92 93 94 95 96 97 98 118 33 35 36 37 38 76 120 122 127 130 131 46 61 66 74 99 133 134 1 2 5 6 7 12 13 41 42 43 70 71 77 78 79 80 81 82 101 103 106 107 108; do 
    bash /public/home/xiehaibing7/caominghao_v11/c12.convert_normal/merge_male_pop_pos_flip_HPC.sh \
        -t ${trait} \
        -m male_pos_flip \
        -i ${INPUT_BASE} \
        -o ${OUTPUT_BASE} \
        -y 
done
```
* 提交任务的模板,female population + negative flip
```shell
#!/bin/bash
# 创建执行脚本 submit_jobs.sh
# 定义输入输出路径
INPUT_BASE="/public/home/xiehaibing7/caominghao_v11/11.male_nega_flip_parquet_data"
OUTPUT_BASE="/public/home/xiehaibing7/caominghao_v11/d12.convert_normal"

# 处理male positive flip数据
for trait in 3 9 10 14 15 16 17 18 19 25 26 59 60 63 124 139; do 
    bash /public/home/xiehaibing7/caominghao_v11/c12.convert_normal/merge_male_pop_pos_flip_HPC.sh \
        -t ${trait} \
        -m male_pos_flip \
        -i ${INPUT_BASE} \
        -o ${OUTPUT_BASE} \
        -y 
done
```
* 提交任务的模板,female population + positive flip
```shell
#!/bin/bash
# 创建执行脚本 submit_jobs.sh
# 定义输入输出路径
INPUT_BASE="/public/home/xiehaibing7/caominghao_v11/10.male_pos_flip_parquet_data"
OUTPUT_BASE="/public/home/xiehaibing7/caominghao_v11/c12.convert_normal"

# 处理male positive flip数据
for trait in 3 9 10 14 15 16 17 18 19 25 26 59 60 63 124 139; do 
    bash /public/home/xiehaibing7/caominghao_v11/c12.convert_normal/merge_male_pop_pos_flip_HPC.sh \
        -t ${trait} \
        -m male_pos_flip \
        -i ${INPUT_BASE} \
        -o ${OUTPUT_BASE} \
        -y 
done
```
