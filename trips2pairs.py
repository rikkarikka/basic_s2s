import sys

with open(sys.argv[1]) as f:
  fn = sys.argv[1].split("/")[-1]
  with open("data/"+fn,'w') as g:
    for l in f:
      spl = l.strip().split("\t")
      if all([x.strip() for x in spl]):
        for i in range(len(spl)-1):
          g.write(spl[i]+"\t"+spl[i+1]+"\n")
