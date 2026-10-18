import torch, struct
inv32 = 1.0/(1e6 ** (torch.arange(0,128,2,dtype=torch.float)/128))
open("inv_torch.txt","w").write("\n".join(str(struct.unpack('<I', struct.pack('<f', float(v)))[0]) for v in inv32))
