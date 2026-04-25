#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import sys
import argparse
import io


class FlowListDumper(yaml.SafeDumper):
    """
    自定义YAML Dumper，将所有列表转换为flow format
    """
    def represent_sequence(self, tag, sequence, flow_style=None):
        # 对于所有列表，强制使用flow style
        return super().represent_sequence(tag, sequence, flow_style=True)


class SelectiveFlowDumper(yaml.SafeDumper):
    """
    自定义YAML Dumper，只将简单列表转换为flow format
    简单列表定义：所有元素都是标量类型（非列表、非字典）
    """
    def represent_sequence(self, tag, sequence, flow_style=None):
        # 检查列表是否是简单列表
        is_simple = all(
            not isinstance(item, (list, dict))
            for item in sequence
        )
        # 如果是简单列表，使用flow style；否则使用block style
        return super().represent_sequence(tag, sequence, flow_style=is_simple)


class CustomDumper(yaml.SafeDumper):
    """
    更智能的自定义YAML Dumper：
    1. 对于空列表，使用flow format
    2. 对于只有一个元素的列表，使用flow format
    3. 对于包含简单元素的短列表，使用flow format
    4. 对于复杂列表或长列表，使用block format
    """
    def represent_sequence(self, tag, sequence, flow_style=None):
        # 自定义flow style逻辑
        if not sequence:  # 空列表
            flow_style = True
        elif len(sequence) == 1:  # 只有一个元素的列表
            flow_style = True
        elif len(sequence) <= 5:  # 短列表（<=5个元素）
            # 检查是否所有元素都是简单类型
            is_simple = all(
                not isinstance(item, (list, dict))
                for item in sequence
            )
            flow_style = is_simple
        else:
            flow_style = False  # 长列表使用block style
        
        return super().represent_sequence(tag, sequence, flow_style=flow_style)


def yaml_block_to_flow(input_file: str, output_file: str = None, indent: int = 2, style: str = "selective") -> None:
    """
    将YAML文件中的block format数组转换为flow format
    
    Args:
        input_file: 输入YAML文件路径
        output_file: 输出YAML文件路径，如果为None则输出到控制台
        indent: 缩进空格数
        style: 转换风格：
            - "all": 将所有列表转换为flow format
            - "selective": 只将简单列表转换为flow format
            - "smart": 智能转换，根据列表内容和长度选择格式
    """
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 选择Dumper
        if style == "all":
            dumper_class = FlowListDumper
        elif style == "selective":
            dumper_class = SelectiveFlowDumper
        elif style == "smart":
            dumper_class = CustomDumper
        else:
            raise ValueError(f"不支持的转换风格: {style}")
        
        # 配置Dumper
        dumper_class.ignore_aliases = lambda *args: True  # 禁用别名
        
        # 创建一个临时缓冲区来捕获YAML输出
        buffer = io.StringIO()
        
        # 序列化到临时缓冲区
        yaml.dump(
            data,
            buffer,
            Dumper=dumper_class,
            allow_unicode=True,
            indent=indent,
            sort_keys=False  # 保持键的顺序
        )
        
        # 获取序列化结果
        yaml_str = buffer.getvalue()
        
        # 写入输出文件或控制台
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
            print(f"转换完成！已写入文件: {output_file}")
        else:
            print(yaml_str)
            
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"错误：YAML解析失败 - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将YAML文件中的block format数组转换为flow format")
    parser.add_argument("input_file", help="输入YAML文件路径")
    parser.add_argument("-o", "--output", help="输出YAML文件路径，默认为控制台输出")
    parser.add_argument("-i", "--indent", type=int, default=2, help="缩进空格数，默认为2")
    parser.add_argument(
        "-s", "--style", 
        choices=["all", "selective", "smart"], 
        default="selective", 
        help="转换风格：all（所有列表）、selective（仅简单列表）、smart（智能选择），默认为selective"
    )
    
    args = parser.parse_args()
    yaml_block_to_flow(args.input_file, args.output, args.indent, args.style)