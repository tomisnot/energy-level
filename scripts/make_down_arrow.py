# -*- coding: utf-8 -*-
"""生成 src/ui/assets/down_arrow.png（QComboBox 下拉三角，中性灰 #6b7280）。

纯 Python PNG 编码（zlib + CRC，无 Qt/外部依赖）——确定性强、可在无
QApplication 环境运行。三角：12x8，底边 y=1 宽 10px，顶点 y=7 居中，
1px 边缘做 50% 半透明抗锯齿。
"""
import os
import struct
import zlib

W, H = 12, 8
R, G, B = 0x6b, 0x72, 0x80        # COL_TEXT_DIM 中性灰
OUT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))),   # 仓库根
    'src', 'ui', 'assets', 'down_arrow.png')


def _pixel(x, y):
    """返回 (r, g, b, a)；透明返回 None。三角：顶点 (5.5, 7)，底边 y=1（x 1..10）。"""
    t = (y - 1.0) / 6.0            # 0=底边 → 1=顶点
    if not (0.0 <= t <= 1.0):
        return None
    half = (1.0 - t) * 4.5         # 底边半宽 4.5 → 顶点 0
    dx = abs(x - 5.5)
    if dx <= half - 0.45:          # 实心
        return (R, G, B, 255)
    if dx <= half + 0.45:          # 边缘 50% 抗锯齿
        return (R, G, B, 128)
    return None


def _png_chunk(tag, data):
    c = struct.pack('>I', len(data)) + tag + data
    c += struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    return c


def make_png():
    raw = bytearray()
    for y in range(H):
        raw.append(0)                              # filter: None
        for x in range(W):
            p = _pixel(x, y)
            raw += bytes(p) if p else bytes((0, 0, 0, 0))
    ihdr = struct.pack('>IIBBBBB', W, H, 8, 6, 0, 0, 0)   # 8bit RGBA
    png = (b'\x89PNG\r\n\x1a\n'
           + _png_chunk(b'IHDR', ihdr)
           + _png_chunk(b'IDAT', zlib.compress(bytes(raw), 0))
           + _png_chunk(b'IEND', b''))
    return png


if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    data = make_png()
    with open(OUT, 'wb') as f:
        f.write(data)
    print(f'{OUT} 生成: {len(data)} 字节')
