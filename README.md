# advent-of-code-2022
Just for fun, ugly solutions essentially guaranteed.

https://adventofcode.com/

Clicks links below for solutions.

[Day 1](##Day-1)

[Day 2](##Day-2)

[Day 3](##Day-3)

[Day 4](##Day-4)

[Day 5](##Day-5)

[Day 6](##Day-6)

[Day 7](##Day-7)

[Day 8](##Day-8)

[Day 9](##Day-9)

[Day 10](##Day-10)

[Day 11](##Day-11)


```python
# import used thus far, trying to keep 
# to minimum, without needing to write
# excessive code...

import re
from copy import deepcopy
from math import lcm
from functools import reduce
```

## Day 1


```python
with open('data/day1.txt') as file:
    data = file.read()

summed = []
current = 0

for cal in data.splitlines():
    if cal !="":
        current+=int(cal)
    else:
        summed.append(current)
        current=0

summed.sort()

print(summed[-1])
print(sum(summed[-3:]))
```

    74394
    212836


## Day 2


```python
with open('data/day2.txt') as file:
    input = file.read()

input = input.replace('A', 'X').replace('B','Y').replace('C','Z')
input = [i.split(' ') for i in input.splitlines()]


picks = {'X':1, 'Y':2, 'Z':3}
losing = {'X':'Z', 'Y':'X', 'Z':'Y'}
winning = {'Z':'X', 'X':'Y', 'Y':'Z'}
```


```python
score = 0
for draw in input:
    score += picks[draw[1]]
    if draw[0]==draw[1]:
        score+=3
        continue
    elif draw[1]=='X':
        if draw[0]=='Z':
            score+=6
        else:
            continue
    elif draw[1]=='Y':
        if draw[0]=='X':
            score+=6
        else:
            continue
    elif draw[1]=='Z':
        if draw[0]=='Y':
            score+=6

# question 1
print(score)
```

    11063



```python
input2 = []
for draw in input:
    temp = [draw[0]]
    if draw[1]=='Y':
        temp.append(draw[0])
    elif draw[1]=='X':
        temp.append(losing[draw[0]])
    else:
        temp.append(winning[draw[0]])
    input2.append(temp)


score = 0

for draw in input2:
    score += picks[draw[1]]
    if draw[0]==draw[1]:
        score+=3
        continue
    elif draw[1]=='X':
        if draw[0]=='Z':
            score+=6
        else:
            continue
    elif draw[1]=='Y':
        if draw[0]=='X':
            score+=6
        else:
            continue
    elif draw[1]=='Z':
        if draw[0]=='Y':
            score+=6

# question 2
print(score)
```

    10349


## Day 3


```python
with open('data/day3.txt') as file:
    input = file.read().splitlines()
```


```python
letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

common = {}

for i in input:
    a = i[:(len(i)) // 2]
    b = i[(len(i)) // 2:]
    for char in set(a).intersection(b):
        if char in common:
            common[char]+=1
        else:
            common[char]=1

score = 0
for key in common.keys():
    score += (letters.index(key)+1) * common[key]
score
```




    7446




```python
common = {}

for i in range(len(input)//3):
    a = input[i*3]
    b = input[i*3+1]
    c = input[i*3+2]

    group = set(a).intersection(set(b)).intersection(set(c))

    if list(group)[0] in common:
        common[list(group)[0]]+=1
    else:
        common[list(group)[0]]=1

score = 0
for key in common.keys():
    score += (letters.index(key)+1) * common[key]
score
```




    2646



## Day 4


```python
with open('data/day4.txt') as file:
    input = file.read().splitlines()

input = [i.split(',') for i in input]
```


```python
count=0

for i in input:
    a = [int(k) for k in i[0].split('-')]
    b = [int(k) for k in i[1].split('-')]

    if a[0]==b[0] or a[1]==b[1]:
        count+=1
    elif a[0]<b[0]:
        if a[1]>b[1]:
            count+=1
    elif b[1]>a[1]:
        count+=1

count
```




    487




```python
count=0

for i in input:
    a = [int(k) for k in i[0].split('-')]
    b = [int(k) for k in i[1].split('-')]

    if max(b) >= a[0] >= min(b):
        count += 1
    elif max(b) >= a[1] >= min(b):
        count += 1
    elif max(a) >= b[0] >= min(a):
        count += 1
    elif max(a) >= b[1] >= min(a):
        count += 1

count
```




    849



## Day 5


```python
with open('data/day5.txt') as file:
    input = file.read()

crates = [list((input.split('1   2')[0].replace('\n',''))[1+4*i::35].replace(' ','')) for i in range(9)]
crates2 = deepcopy(crates)

moves = [re.findall(pattern=r'[0-9]+', string=m) for m in input.split('8   9')[1].splitlines()[2:]]
```


```python
for m in moves:
    m = [int(i)-1 for i in m]
    for _ in range(m[0]+1):
        crates[m[2]].insert(
                0,
                crates[m[1]].pop(0))
                
# question 1
print(''.join([c[0] for c in crates]))
```

    QNHWJVJZW



```python
for m in moves:
    m = [int(i)-1 for i in m]
    crates2[m[2]] = crates2[m[1]][:m[0]+1]+crates2[m[2]]
    crates2[m[1]] = crates2[m[1]][m[0]+1:]

# question 2
''.join([c[0] for c in crates2])
```




    'BPCZJLFJW'



## Day 6


```python
with open('data/day6.txt') as file:
    input = file.read()
```


```python
for i in range(4, len(input)):
    if len(set(input[i-4:i]))==4:
        print(i)
        break

for i in range(14, len(input)):
    if len(set(input[i-14:i]))==14:
        print(i)
        break
```

    1848
    2308


## Day 7


```python
with open('data/day7.txt') as file:
    input = file.read()

dir = {'/' : {}}
path = []

for l in input.splitlines():
    args = l.split(' ')
    if '$' in l:
        if 'cd' in l:
            if '..' in l:
                path.pop()
            elif '/' in l:
                path = [dir['/']]
            else:
                path.append(path[-1][args[2]])
    else:
        if 'dir' in l:
            path[-1][args[1]] = {}  
        else:
            path[-1][args[1]] = int(args[0])

filesize = {}

def recursion(base_dir, dir_name):
    size = 0
    for d,subdir in base_dir.items():
        if type(subdir) == dict:
            key = dir_name + '/' + d if dir_name else '/'
            filesize[key] = recursion(subdir, key)
            size += filesize[key]
        else:
            size += subdir
    return size

recursion(dir, '')

# question1
print(sum(fs for fs in filesize.values() if fs <= 100000))

# question2
print(min(fs for fs in filesize.values() if fs > filesize['/'] - 40000000))
```

    1443806
    942298


## Day 8


```python
with open('data/day8.txt') as file:
    input = file.read()

matrix = []
empties = []
for line in input.splitlines():
    matrix.append([int(i) for i in list(line)])
    empties.append([0]*len(line))

for row, empty in zip(matrix, empties):
    l, r = 0, len(row)-1
    l_max, r_max = row[l], row[r]
    empty[l], empty[r] = 1, 1
    while l<r:
        if l_max<r_max:
            l+=1
            if row[l]>l_max:
                empty[l]=1
                l_max = row[l]
        else:
            r-=1
            if row[r]>r_max:
                empty[r]=1
                r_max = row[r]

# turn 90 degrees and do again
matrix = [list(l) for l in zip(*matrix[::-1])]
empties = [list(l) for l in zip(*empties[::-1])]

for row, empty in zip(matrix, empties):
    l, r = 0, len(row)-1
    l_max, r_max = row[l], row[r]
    empty[l], empty[r] = 1, 1
    while l<r:
        if l_max<r_max:
            l+=1
            if row[l]>l_max:
                empty[l]=1
                l_max = row[l]
        else:
            r-=1
            if row[r]>r_max:
                empty[r]=1
                r_max = row[r]

# question 1
print(sum(map(sum, empties)))


max_range = 0
for i in range(len(matrix)):
    for j in range(len(matrix[0])):
        tree = matrix[i][j]
        up, down, right, left = 0, 0, 0, 0

        x=i
        while x>0:
            x-=1
            up+=1
            if matrix[x][j] >= tree:
                break
        x=i
        while x<len(matrix)-1:
            x+=1
            down+=1
            if matrix[x][j] >= tree:
                break
        x=j
        while x<len(matrix[0])-1:
            x+=1
            right+=1
            if matrix[i][x] >= tree:
                break
        x=j
        while x>0:
            x-=1
            left+=1
            if matrix[i][x] >= tree:
                break

        vision = up*down*right*left        
        max_range = max(max_range, vision)

print(max_range)
```

    1708
    504000


## Day 9


```python
with open('data/day9.txt') as file:
    input = file.read()

head = [0, 0]
tail = [0, 0]
visited = []

for step in input.splitlines():
    dir, length = step.split(' ')[0], int(step.split(' ')[1])
    for _ in range(length):
        # move head
        if dir=='R':
            head[0]+=1
        elif dir=='L':
            head[0]-=1
        elif dir=='U':
            head[1]+=1
        elif dir=='D':
            head[1]-=1
        
        # check if diagonal movement
        diag = max(
            abs(head[0]-tail[0]),
            abs(head[0]-tail[0]),
            abs(head[1]-tail[1]),
            abs(head[1]-tail[1])
            ) > 1
        
        # move tail
        if diag:
            if head[0]>tail[0]:
                tail[0]+=1
            if head[0]<tail[0]:
                tail[0]-=1
            if head[1]>tail[1]:
                tail[1]+=1
            if head[1]<tail[1]:
                tail[1]-=1 
        else: 
            if head[0]-tail[0]>1:
                tail[0]+=1
            if head[0]-tail[0]<-1:
                tail[0]-=1
            if head[1]-tail[1]>1:
                tail[1]+=1
            if head[1]-tail[1]<-1:
                tail[1]-=1 

        # check if new tail position
        if tail not in visited:
            visited.append(tail[:])

# question 1
print(len(visited))

tails = [[0, 0] for _ in range(10)]
visited = []

for step in input.splitlines():
    dir, length = step.split(' ')[0], int(step.split(' ')[1])
    for _ in range(length):
        # same approach as above but move tails
        # iteratively refering the previous tail
        if dir=='R':
            tails[0][0]+=1
        elif dir=='L':
            tails[0][0]-=1
        elif dir=='U':
            tails[0][1]+=1
        elif dir=='D':
            tails[0][1]-=1
        
        for i in range(1, 10):
            diag = max(
                abs(tails[i-1][0]-tails[i][0]),
                abs(tails[i-1][0]-tails[i][0]),
                abs(tails[i-1][1]-tails[i][1]),
                abs(tails[i-1][1]-tails[i][1])
                ) > 1
            
            if diag:
                if tails[i-1][0]>tails[i][0]:
                    tails[i][0]+=1
                if tails[i-1][0]<tails[i][0]:
                    tails[i][0]-=1
                if tails[i-1][1]>tails[i][1]:
                    tails[i][1]+=1
                if tails[i-1][1]<tails[i][1]:
                    tails[i][1]-=1 
            else: 
                if tails[i-1][0]-tails[i][0]>1:
                    tails[i][0]+=1
                if tails[i-1][0]-tails[i][0]<-1:
                    tails[i][0]-=1
                if tails[i-1][1]-tails[i][1]>1:
                    tails[i][1]+=1
                if tails[i-1][1]-tails[i][1]<-1:
                    tails[i][1]-=1 


        if tails[-1] not in visited:
            visited.append(tails[i][:])

print(len(visited))
```

    5695
    2434


## Day 10


```python
with open('data/day10.txt') as file:
    input = file.read()
```


```python
sum = 1
i = 1

points = [20, 60, 100, 140, 180, 220]
total_score = 0


screen = [[" "] * 40 for _ in range(6)]

def draw(cycle, strength):
    cycle-=1
    row = cycle//40
    col = cycle%40
    if abs(strength-col)<2:
        screen[row][col]='#'    

for line in input.splitlines():
    draw(cycle=i, strength=sum)
    i+=1
    
    if i in points:
        total_score += points.pop(0) * sum

    if num:=re.findall(pattern=r'-*[0-9]+', string=line):
        draw(cycle=i, strength=sum)
        i+=1
        sum+=int(num[0])

    if i in points:
        total_score += points.pop(0) * sum


print(total_score)
for row in screen:
    print(''.join(row))
```

    13440
    ###  ###  ####  ##  ###   ##  ####  ##  
    #  # #  #    # #  # #  # #  #    # #  # 
    #  # ###    #  #    #  # #  #   #  #  # 
    ###  #  #  #   # ## ###  ####  #   #### 
    #    #  # #    #  # # #  #  # #    #  # 
    #    ###  ####  ### #  # #  # #### #  # 


## Day 11


```python
class Monkey:
  """
  Fully fledged item throwing
  and inspecting monkeys.

  Create using monkey parser
  """
  def __init__(
      self,
      number:int,
      items:list,
      operation:str,
      reduction:str,
      test:int,
      friends:list
      ):
    self.number = number
    self.items = items
    self.operation = operation
    self.reduction = reduction
    self.test = test
    self.friends = friends
    self.inspections = 0

  def _inspect(self) -> None:
    """
    Apply their inspection function to
    keep track of our worry, also log
    number of inspections
    """
    for i in range(len(self.items)):
      # their operation
      self.items[i]=eval(
        self.operation.replace('old', str(self.items[i])))
      # function to reduce worry
      self.items[i] = eval(
        self.reduction.replace('old', str(self.items[i])))
      # keep count of inspections
      self.inspections += 1

  def _test(self) -> dict:
    """
    Apply their test to decide where to throw
    item, retur dict of items to throw
    """
    to_throw = {f:[] for f in self.friends}
    for i in self.items:
      if i % self.test == 0:
        to_throw[self.friends[0]].append(i)
      else:
        to_throw[self.friends[1]].append(i)
    self.items = []
    return to_throw

  def tour(self) -> dict:
    """
    Wrap inspection and test, return dict
    of items to throw
    """
    # one round for one monkey
    self._inspect()
    to_throw = self._test()
    return to_throw


class Jungle(Monkey):
  """
  Jungle class keeping track of entire
  pack of monkeys. 
  """
  def __init__(
      self,
      monkeys:list[Monkey]
      ):
    self.monkeys = monkeys

  def tour(
      self,
      tours: int=1,
      log: bool=True
      ) -> None:
    """
    Have entire pack of monkeys do their
    item inspections and throw items to 
    pertinent monkey friends.

    Allows for "pretty" logging.
    """
    for t in range(tours):
      for monkey in self.monkeys:
        to_throw = monkey.tour()
        for friend in to_throw.keys():
          self.monkeys[friend].items += to_throw[friend]

      if log:
        print('Round %s' % str(t+1))
        for monkey in self.monkeys: 
          print('Monkey %s :' % monkey.number, monkey.items)

  def monkey_business(self) -> int:
    """
    Get current of monkey business.
    """
    inspections = sorted([monkey.inspections for monkey in self.monkeys])
    return inspections[-2]*inspections[-1]
      

def monkey_parser(monkey_txt: list):
  """
  Function to parse text for induvidual monkeys
  and return attributes
  """
  attributes = {}
  # get monkey number
  attributes['number'] = \
    int(''.join([i for i in list(monkey_txt[0]) if i in '0123456789']))
  # get list of items
  attributes['items'] = \
    [int(i) for i in monkey_txt[1].split(': ')[1].split(', ')]
  # get worry model
  attributes['operation'] = \
    monkey_txt[2].split('= ')[1]
  # get test
  attributes['test'] = \
    int(monkey_txt[3].split()[-1])
  # get friends
  attributes['friends'] = [
    int(monkey_txt[-2].split()[-1]), int(monkey_txt[-1].split()[-1])]

  return attributes


def jungle_parser(jungle_txt: list[str], question: int=1):
  """
  Function to parse text for entire jungle 
  and return attributes
  """
  # in questions 1 worry is reduced as
  # worry = worry // 3 
  if question == 1:
    reduction = 'old // 3'
  # in questions 2 we can remove the lcm 
  # from the worry levels. all tests are primes
  # thus if it is divisible by the lcm of them 
  # it has already passed by all monkeys and
  # is back in the starting position
  else:
    tests = [monkey_parser(monkey_txt)['test'] for monkey_txt in jungle_txt]
    test_lcm = reduce(lcm, tests)
    reduction = 'old % {}'.format(test_lcm)
    
  monkeys = []
  for monkey_txt in jungle_txt:
    monkeys.append(Monkey(**monkey_parser(monkey_txt), reduction=reduction))
  return monkeys
```


```python
with open('data/day11.txt') as file:
    input = file.read()
    
jungle_txt = [i.splitlines() for i in input.split('\n\n')]
```


```python
jungle = Jungle(jungle_parser(jungle_txt=jungle_txt, question=1))
jungle.tour(tours=20, log=False)
jungle.monkey_business()
```




    101436




```python
jungle = Jungle(jungle_parser(jungle_txt=jungle_txt, question=2))
jungle.tour(tours=10_000, log=False)
jungle.monkey_business()
```




    19754471646


