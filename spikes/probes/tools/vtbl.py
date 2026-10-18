import re, sys
hdr = open(sys.argv[1], encoding='utf-8', errors='replace').read()
want = sys.argv[3].split(',')
for iface in sys.argv[2].split(','):
    m = re.search(r'typedef struct %sVtbl\s*\{(.*?)END_INTERFACE' % re.escape(iface), hdr, re.S)
    if not m:
        print(iface, 'NOT FOUND'); continue
    names = re.findall(r'DECLSPEC_XFGVIRT\(\s*\w+\s*,\s*(\w+)\s*\)', m.group(1))
    print(iface, len(names), ' '.join('%d:%s' % (i, n) for i, n in enumerate(names) if n in want or want == ['*']))
