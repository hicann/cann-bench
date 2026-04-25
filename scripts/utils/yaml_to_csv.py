#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将单个YAML文件转换为CSV文件的工具脚本
支持指定输入/输出文件路径，正确处理边界场景
"""

import yaml
import csv
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union


def format_array_value(value: Any) -> str:
    """
    格式化数组值，确保正确处理各种边界情况
    
    Args:
        value: 要格式化的值
        
    Returns:
        格式化后的字符串
    """
    if value is None:
        return "[]"
    elif isinstance(value, list):
        # 使用json.dumps确保正确处理嵌套数组和特殊字符
        return json.dumps(value)
    elif isinstance(value, (str, int, float, bool)):
        # 如果是单个值，将其包装为数组
        return json.dumps([value])
    else:
        # 其他类型转换为字符串并包装为数组
        return json.dumps([str(value)])


def format_dict_value(value: Any) -> str:
    """
    格式化字典值，确保正确处理各种边界情况
    
    Args:
        value: 要格式化的值
        
    Returns:
        格式化后的字符串
    """
    if value is None:
        return "{}"
    elif isinstance(value, dict):
        # 使用json.dumps确保正确处理嵌套字典和特殊字符
        return json.dumps(value)
    else:
        # 其他类型转换为字符串
        return str(value)


def process_yaml_field(field_name: str, value: Any) -> str:
    """
    根据字段名称和类型进行适当的格式化
    
    Args:
        field_name: 字段名称
        value: 字段值
        
    Returns:
        格式化后的字段值
    """
    # 处理数组类型字段
    array_fields = ['input_shape', 'dtype', 'value_range']
    if field_name in array_fields:
        return format_array_value(value)
    
    # 处理字典类型字段
    dict_fields = ['attrs']
    if field_name in dict_fields:
        return format_dict_value(value)
    
    # 处理字符串类型字段（可能包含特殊字符）
    if isinstance(value, str):
        # 替换换行符为空格，确保CSV格式正确
        return value.replace('\n', ' ').replace('\r', '')
    
    # 处理None值
    if value is None:
        return ''
    
    # 默认转换为字符串
    return str(value)


def yaml_to_csv(input_file: str, output_file: Optional[str] = None) -> bool:
    """
    将单个YAML文件转换为CSV文件
    
    Args:
        input_file: 输入YAML文件路径
        output_file: 输出CSV文件路径，如果为None则自动生成
        
    Returns:
        转换成功返回True，否则返回False
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误：输入文件 {input_file} 不存在", file=sys.stderr)
            return False
        
        # 读取YAML文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 检查是否包含cases字段
        cases = data.get('cases', [])
        if not cases:
            print(f"警告：YAML文件 {input_file} 中没有找到cases字段", file=sys.stderr)
            return False
        
        # 确定输出文件路径
        if output_file is None:
            output_file = os.path.splitext(input_file)[0] + '.csv'
        
        # 确定所有可能的字段
        all_fields = set()
        for case in cases:
            all_fields.update(case.keys())
        
        # 定义标准字段顺序
        standard_fields = ['operator', 'case_id', 'input_shape', 'dtype', 'attrs', 'value_range', 'baseline_perf_us', 'note']
        
        # 合并字段：标准字段在前，其他字段在后
        csv_fields = []
        for field in standard_fields:
            if field in all_fields:
                csv_fields.append(field)
                all_fields.remove(field)
        
        # 添加剩余的自定义字段
        csv_fields.extend(sorted(all_fields))
        
        # 写入CSV文件
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            
            # 写入表头
            writer.writeheader()
            
            # 写入每行数据
            for case in cases:
                # 格式化每个字段
                formatted_case = {}
                for field, value in case.items():
                    formatted_case[field] = process_yaml_field(field, value)
                
                # 确保所有字段都存在（缺失字段填充空字符串）
                row = {field: formatted_case.get(field, '') for field in csv_fields}
                
                writer.writerow(row)
        
        print(f"成功：已将 {input_file} 转换为 {output_file}（共 {len(cases)} 个用例）")
        return True
        
    except yaml.YAMLError as e:
        print(f"错误：YAML文件解析失败 - {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"错误：转换失败 - {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数，处理命令行参数
    """
    parser = argparse.ArgumentParser(description="将单个YAML文件转换为CSV文件")
    parser.add_argument("input_file", help="输入YAML文件路径")
    parser.add_argument("-o", "--output", help="输出CSV文件路径，默认与输入文件同名但扩展名为.csv")
    
    args = parser.parse_args()
    
    if yaml_to_csv(args.input_file, args.output):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()