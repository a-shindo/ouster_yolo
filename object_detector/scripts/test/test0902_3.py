import sys

var1=sys.argv[1]
try:
    try:
        var2=sys.argv[2]
    except IndexError:
        var2="test"
except:
    print("here")
print(sys.argv[0])
print(var1)
print(var2)