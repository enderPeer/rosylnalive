"""Reads Khronos vk.xml (as data) and emits, for a list of structs:
  --fsharp : F# [<Struct>] declarations (used for the large, boring structs)
  --layout : an F# table of C-ABI (Windows/Linux x64) sizes and member offsets,
             which the console app compares against the CLR's actual layout.
Usage: python -I vkxml_layout.py <vk.xml> (--fsharp|--layout) Struct1,Struct2,...
"""
import sys, xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
mode = sys.argv[2]
wanted = sys.argv[3].split(',')

consts = {}
for e in root.iter('enum'):
    v = e.get('value')
    if e.get('name') and v and v.isdigit():
        consts[e.get('name')] = int(v)

types = {}
for t in root.find('types'):
    n = t.get('name') or (t.find('name').text if t.find('name') is not None else None)
    if n and t.get('api') != 'vulkansc':
        types.setdefault(n, t)

PRIM = {  # C type -> (size, align, F# type)
    'uint8_t': (1, 1, 'byte'), 'int32_t': (4, 4, 'int32'), 'uint32_t': (4, 4, 'uint32'),
    'uint64_t': (8, 8, 'uint64'), 'float': (4, 4, 'float32'), 'size_t': (8, 8, 'unativeint'),
    'char': (1, 1, 'byte'),
    'VkBool32': (4, 4, 'VkBool32'), 'VkDeviceSize': (8, 8, 'uint64'),
    'VkSampleCountFlags': (4, 4, 'uint32'), 'VkFlags': (4, 4, 'uint32'),
}

def members(name):
    out = []
    for m in types[name].findall('member'):
        if m.get('api') == 'vulkansc':
            continue
        ty = m.find('type').text
        nm = m.find('name').text
        ptr = '*' in ''.join(x for x in m.itertext() if x not in (ty, nm))
        tail = (m.find('name').tail or '').strip()
        count = 1
        if tail.startswith('['):
            enum = m.find('enum')
            count = consts[enum.text] if enum is not None else int(tail.strip('[]'))
        out.append((nm, ty, ptr, count))
    return out

def type_info(ty, ptr):
    if ptr:
        return (8, 8, 'nativeint')
    if ty in PRIM:
        return PRIM[ty]
    t = types[ty]
    cat = t.get('category')
    if cat == 'enum':
        return (4, 4, ty)
    if cat == 'bitmask':
        return (4, 4, 'uint32')
    if cat == 'struct':
        s, a = layout(ty)[:2]
        return (s, a, ty)
    raise Exception('unknown type ' + ty)

def layout(name):
    off, align, offs = 0, 1, []
    for nm, ty, ptr, count in members(name):
        s, a, _ = type_info(ty, ptr)
        off = (off + a - 1) // a * a
        offs.append((nm, off))
        off += s * count
        align = max(align, a)
    size = (off + align - 1) // align * align
    return size, align, offs

FIXED = {('byte', 256): 'FixedBytes256', ('byte', 16): 'FixedBytes16', ('uint32', 3): 'FixedUInt32x3',
         ('uint32', 2): 'FixedUInt32x2', ('float32', 2): 'FixedFloat32x2'}

if mode == '--fsharp':
    for n in wanted:
        print('[<Struct; StructLayout(LayoutKind.Sequential)>]')
        print('type %s =' % n)
        for nm, ty, ptr, count in members(n):
            fs = type_info(ty, ptr)[2]
            if count > 1:
                fs = FIXED[(fs, count)]
            print('    val mutable %s: %s' % (nm, fs))
        print()
elif mode == '--layout':
    print('    [')
    for n in wanted:
        size, _, offs = layout(n)
        print('        "%s", %d, [ %s ]' % (n, size, '; '.join('"%s", %d' % o for o in offs)))
    print('    ]')
