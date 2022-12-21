# advent-of-code-2022
Just for fun, ugly solutions essentially guaranteed.

https://adventofcode.com/

### Finished days:

[Day 1](#day-1)

[Day 2](#day-2)

[Day 3](#day-3)

[Day 4](#day-4)

[Day 5](#day-5)

[Day 6](#day-6)

[Day 7](#day-7)

[Day 8](#day-8)

[Day 9](#day-9)

[Day 10](#day-10)

[Day 11](#day-11)

[Day 12](#day-12)

[Day 13](#day-13)

[Day 14](#day-14)

[Day 15](#day-15)

[Day 16](#day-16)

[Day 17](#day-17)

[Day 18](#day-18)

[Day 19](#day-19)

[Day 20](#day-20)

[Day 21](#day-21)



```python
# import used thus far, trying to keep 
# to minimum, without needing to write
# excessive code...

import re
from copy import deepcopy
from math import lcm
from functools import reduce
from random import sample
from collections import deque
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
    for k in range(len(matrix[0])):
        tree = matrix[i][k]
        up, down, right, left = 0, 0, 0, 0

        x=i
        while x>0:
            x-=1
            up+=1
            if matrix[x][k] >= tree:
                break
        x=i
        while x<len(matrix)-1:
            x+=1
            down+=1
            if matrix[x][k] >= tree:
                break
        x=k
        while x<len(matrix[0])-1:
            x+=1
            right+=1
            if matrix[i][x] >= tree:
                break
        x=k
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
        
        # check if diagonal movement,
        # this happens when manhattan
        # distance is > 2.
        diag = sum([
            abs(head[0]-tail[0]),
            abs(head[0]-tail[0]),
            abs(head[1]-tail[1]),
            abs(head[1]-tail[1])
            ]) > 2
        
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

        # add tail position
        visited.append(tuple(tail[:]))

# question 1
print(len(set(visited)))

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
            diag = sum([
                abs(tails[i-1][0]-tails[i][0]),
                abs(tails[i-1][0]-tails[i][0]),
                abs(tails[i-1][1]-tails[i][1]),
                abs(tails[i-1][1]-tails[i][1])
                ]) > 2
            
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


        visited.append(tuple(tails[i][:]))

print(len(set(visited)))
```

    5764
    2616


## Day 10


```python
with open('data/day10.txt') as file:
    input = file.read()
```


```python
total = 1
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
    draw(cycle=i, strength=total)
    i+=1
    
    if i in points:
        total_score += points.pop(0) * total

    if num:=re.findall(pattern=r'-*[0-9]+', string=line):
        draw(cycle=i, strength=total)
        i+=1
        total+=int(num[0])

    if i in points:
        total_score += points.pop(0) * total


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
    Apply inspection function to
    keep track of worry, also log
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
    items, return dict of items to throw
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


class Jungle():
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

    log allows for "pretty" logging.
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
    Get current amount of monkey business.
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
  # since 
  #   x % n = x % n % n,
  # we can save x % n instead of x
  # with n being the lcm of all tests, 
  # effectively preventing the worry 
  # level of ever exceeding n, without
  # impacting the tests
  else:
    tests = [monkey_parser(monkey_txt)['test'] for monkey_txt in jungle_txt]
    test_lcm = reduce(lcm, tests)
    reduction = 'old % {}'.format(test_lcm)
    
  monkeys = []
  for monkey_txt in jungle_txt:
    monkeys.append(Monkey(**monkey_parser(monkey_txt), reduction=reduction))
  return Jungle(monkeys)

```


```python
with open('data/day11.txt') as file:
    input = file.read()
jungle_txt = [i.splitlines() for i in input.split('\n\n')]
```


```python
jungle = jungle_parser(jungle_txt=jungle_txt, question=1)
jungle.tour(tours=20, log=False)
jungle.monkey_business()
```




    101436




```python
jungle = jungle_parser(jungle_txt=jungle_txt, question=2)
jungle.tour(tours=10_000, log=False)
jungle.monkey_business()
```




    19754471646



## Day 12


```python
def idx(input, row, col):
    """
    Helper function creating an int
    index from a 2d index in a matrix
    """
    return len(input[0])*row + col
    

class Mountain:
    """
    Graph representation of all feasible
    paths of a matrix representation of a
    mountain
    """
    def __init__(
            self,
            matrix: list[list[int]],
            inverse: bool=False
            ):
        # set -1 paths between all points on mountain
        self.edges = [
            [-1 for _ in range(len(matrix)*len(matrix[0]))]
            for _ in range(len(matrix)*len(matrix[0]))
            ]
        # iterate once over input to get all paths
        self._matrix_to_graph(matrix=matrix, inverse=inverse)
        self.v = len(matrix)*len(matrix[0])
        self.visited = []
    

    def _add_edge(
            self,
            u,
            v,
            weight
            ):
        self.edges[u][v] = weight


    def _matrix_to_graph(
            self,
            matrix,
            inverse
            ):
        # inverse changes all elements to negative
        # this allows djikstra to search from end
        # to start instead of inverse... 
        if inverse:
            matrix = [[-x for x in l] for l in matrix]
            
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                
                u, u_value = idx(matrix, row, col), matrix[row][col]


                # check point right of u
                if col+1 < len(matrix[0]):
                    v = idx(matrix, row, col+1)
                    if matrix[row][col+1]-u_value < 2:
                        self._add_edge(u=u, v=v, weight=1)
                    if u_value - matrix[row][col+1] < 2:
                        self._add_edge(u=v, v=u, weight=1)
                # check point below u
                if row+1 < len(matrix):
                    v = idx(matrix, row+1, col)
                    if matrix[row+1][col]-u_value < 2:
                        self._add_edge(u=u, v=v, weight=1)
                    if u_value - matrix[row+1][col] < 2:
                        self._add_edge(u=v, v=u, weight=1)


def dijkstra(
        graph,
        start_vertex,
        ):
    v_num = graph.v
    D = {v:float('inf') for v in range(graph.v)}
    D[start_vertex] = 0

    pq = [start_vertex]

    while pq:
        current_vertex = pq.pop(0)
        graph.visited.append(current_vertex)

        for neighbor in range(graph.v):
            if graph.edges[current_vertex][neighbor] != -1:
                distance = graph.edges[current_vertex][neighbor]
                if neighbor not in graph.visited:
                    old_cost = D[neighbor]
                    new_cost = D[current_vertex] + distance
                    if new_cost < old_cost:
                        pq.append(neighbor)
                        D[neighbor] = new_cost

    graph.v = v_num
    graph.visited = []
    return D

with open('data/day12.txt') as file:
    input = file.read()

ends = []
input = [list(i) for i in input.splitlines()]
for row in range(len(input)):
    for col in range(len(input[0])):
        # see if element is start,
        # save position, relabel
        if input[row][col]=='S':
            start = idx(input, row, col)
            input[row][col]='a'
        # see if element is end,
        # save position, relabel
        elif input[row][col]=='E':
            end = idx(input, row, col)
            input[row][col]='z'
        
        # get potential ends for question 2
        if input[row][col]=='a':
            ends.append(idx(input, row, col))

        # change element to numeric
        input[row][col] = ord(input[row][col])
```


```python
m = Mountain(matrix=input)
D = dijkstra(m, start)
D[end]
```




    339




```python
m = Mountain(matrix=input, inverse=True)
D = dijkstra(m, start_vertex=end)
candidates = []
for d in D.keys():
    if d in ends:
        candidates.append(D[d])
min(candidates)
```




    332



## Day 13


```python
def evaluate(left, right):
    # check if ints, compare
    if type(left)==int and type(right)==int:
        if left == right:
            return 0
        elif left < right:
            return -1
        else:
            return 1

    # create iterators
    try:
        left = iter(left)
    except:
        left = iter((left,))
    try:
        right = iter(right)
    except:
        right = iter((right,))

    # go through iterators
    while True:
        # check if left but not right
        try:
            left1 = next(left)
        except:
            try:
                next(right)
                return -1
            except:
                return 0
        # check right only right
        try:
            right1 = next(right)
        except:
            return 1
        # otherwise recurse
        if comp:=evaluate(left1, right1):
            return comp
```


```python
with open('data/day13.txt') as file:
    input = file.read()

total = 0
i=0
for pair in [i.splitlines() for i in input.split('\n\n')]:
    if evaluate(eval(pair[0]), eval(pair[1])) <= 0:
        total += i + 1
    i+=1
total
```




    5605




```python
div1, div2 = [[2]], [[6]]
count1, count2 = 1, 1
for line in filter(str.strip, input.splitlines()):
    packet = eval(line)
    if evaluate(packet, div1) < 0:
        count1 += 1
    elif evaluate(packet, div2) < 0:
        count2 += 1
count1 * (count1 + count2)
```




    24969



## Day 14


```python
with open('data/day14.txt') as data:
    input = data.read()


rocks = set()
bottom = 0
for line in input.splitlines():
    l = [i.split(',') for i in line.split(' -> ')]
    for start, end in zip(l, l[1:]):
        # get each line separately
        x = sorted([int(start[0]), int(end[0])])
        y = sorted([int(start[1]), int(end[1])])
        # draw entire lines as tuples in set
        for xi in range(x[0], x[1]+1):
            for yi in range(y[0], y[1]+1):
                rocks.add((xi, yi))
                bottom = max(bottom, yi+1)
```


```python
rocks1 = deepcopy(rocks)

volume = 0
filled = False
while True:
    sand = (500, 0)
    while True:
        # sand falls to bottom
        #   ->  figure filled
        if sand[1] >= bottom:
            filled=True
            break
        # sand moves one step down
        if (sand[0], sand[1]+1) not in rocks1:
            sand = (sand[0], sand[1]+1)
            continue
        # sand moves diag left down
        if (sand[0]-1, sand[1]+1) not in rocks1:
            sand = (sand[0]-1, sand[1]+1)
            continue
        # sand moves diag right down
        if (sand[0]+1, sand[1]+1) not in rocks1:
            sand = (sand[0]+1, sand[1]+1)
            continue
        # not failling to bottom and nowhere to move
        #   ->  stay where it is
        rocks1.add(sand)
        volume+=1
        break
    if filled:
        break

volume
```




    994




```python
rocks2 = deepcopy(rocks)

# same process as above but we let sand 
# fall to bottom, only fully break when
# (500, 0 ) has been filled

volume = 0
while (500, 0) not in rocks2:
    sand = (500, 0)
    while True:
        if sand[1] >= bottom:
            break
        if (sand[0], sand[1]+1) not in rocks2:
            sand = (sand[0], sand[1]+1)
            continue
        if (sand[0]-1, sand[1]+1) not in rocks2:
            sand = (sand[0]-1, sand[1]+1)
            continue
        if (sand[0]+1, sand[1]+1) not in rocks2:
            sand = (sand[0]+1, sand[1]+1)
            continue
        break
    rocks2.add(sand)
    volume+=1

volume
```




    26283



## Day 15


```python
with open('data/day15.txt') as data:
    input = data.read()

def manhattan(a, b):
    return abs(a.real - b.real) + abs(a.imag - b.imag)
```


```python
sensors = []
beacons = []
distances = []
for line in input.splitlines():
    line = [int(i) for i in re.findall(r'-*[0-9]+', string=line)]
    sensors.append(line[0] + line[1]*1j)
    beacons.append(line[2] + line[3]*1j)
    distances.append(manhattan(sensors[-1], beacons[-1]))

y = 2000000
occupied = set()
for sensor, distance in zip(sensors, distances):
    if sensor.imag-distance <= y <= sensor.imag+distance:
        spare_distance = int(distance - abs(y-sensor.imag))
        occupied.add(sensor.real)
        for i in range(spare_distance + 1):
            occupied.add(sensor.real - i)
            occupied.add(sensor.real + i)

existing = set([b.real for b in beacons if b.imag==y])

# question 1
len(occupied)-len(existing)
```




    6124805




```python
edge = set()
lower, higher = 0, 4000000

for sensor, distance in zip(sensors, distances):
    if sensor.real<0:
        start = max(-x, -int(distance)-1)
    else:
        start = -int(distance)-1
    for i in range(start, int(distance)+2):
        x = sensor.real + i
        y = sensor.imag + distance + 1 - abs(i)
        if lower <= x <= higher and lower <= y <= higher:
            edge.add(x + y*1j)
        if x >= higher or y <= lower:
            break

    if sensor.imag < 0:
        start = max(-y, -int(distance)-1)
    else:
        start = -int(distance)-1
    for i in range(-int(distance)-1, int(distance)+2):
        x = sensor.real + distance + 1 - abs(i)
        y = sensor.imag + i
        if lower <= x <= higher and lower <= y <= higher:
            edge.add(x + y*1j)
        if x <= lower or y >= higher:
            break

num_sensors = len(sensors)
final = False

for point in edge:
    n = 0
    for sensor, distance in zip(sensors, distances):
        if not manhattan(sensor, point) > distance:
            break
        n += 1
        if n == num_sensors:
            final = point
            break
    if final:
        break

print(final.real * 4000000 + final.imag)
```

    12555527364986.0


## Day 16


```python
with open('data/day16.txt') as data:
    input = data.read()

paths = {}
rates = {}
name_map = {}

int_name=0
for line in input.splitlines():
    name=line.split()[1]
    name_map[name]=int_name
    if int(''.join([n for n in line if n in '1234567890']))>0:
        rates[int_name]=int(''.join([n for n in line if n in '1234567890']))
    paths[int_name]=line.replace('s','').split('valve ')[-1].split(', ')
    int_name+=1

paths = {key:[name_map[i] for i in paths[key]] for key in paths.keys()}
```


```python
# re-purposed from day 12
class Graph:
    def __init__(
            self,
            path_dict: dict,
            ):
        self.edges = [
            [-1]*len(path_dict)
            for _ in range(len(path_dict))
            ]
        self._matrix_to_graph(path_dict=path_dict)
        self.v = len(path_dict)
        self.visited = []

    def _add_edge(
            self,
            u,
            v
            ):
        self.edges[u][v] = 1

    def _matrix_to_graph(
            self,
            path_dict,
            ):
        for u in path_dict.keys():
            for v in path_dict[u]:
                self._add_edge(u=u, v=v)


def path_finder(
        position: int,
        time_left: int,
        to_visit: list,
        simulations: int = 1000,
    ):

    # create random paths by sampling
    paths = [
        list(sample(to_visit, k=len(to_visit)))
        for _ in range(simulations)]

    # try random paths, keep the first move
    # of the best path tried
    top_value = - 10_000
    for path in paths:
        value = 0
        first_character = path[0]
        time_left_, position_ = time_left, position
        while path and time_left_>0:
            next_position = path.pop(0)
            time_left_ -= distances[position_][next_position]+1
            value += max(time_left_*rates[next_position], 0)
            position_ = next_position

        if value >= top_value:
            best_choice = first_character
            top_value = value

    # return first move of best path tried     
    return best_choice


graph = Graph(path_dict=paths)
distances = {key:dijkstra(graph=graph, start_vertex=key) for key in paths.keys()}
```


```python
simulation_results = []
for _ in range(10):
    time_left = 30
    to_visit = list(rates.keys())
    position = name_map['AA']
    total=0

    while time_left>0 and to_visit:    

        best_choice = path_finder(
            position=position,
            time_left=time_left,
            to_visit=to_visit)

        time_left -= distances[position][best_choice]+1
        total += max(0, time_left*rates[best_choice])
        position = to_visit.pop(to_visit.index(best_choice))

    simulation_results.append(total)

print(max(simulation_results))
```

    1716



```python
simulation_results = []
for _ in range(10):
    to_visit = list(rates.keys())
    position1, position2 = name_map['AA'], name_map['AA']
    time_left1, time_left2 = 26, 26
    total=0

    while time_left1>0 or time_left2>0 and to_visit:

        # part for elf

        best_choice = path_finder(
            position=position1,
            time_left=time_left1,
            to_visit=to_visit)

        time_left1 -= distances[position1][best_choice]+1
        total += max(0, time_left1*rates[best_choice])
        position1 = to_visit.pop(to_visit.index(best_choice))

        if not to_visit:
            break

        # part for elephant
        
        best_choice = path_finder(
            position=position2,
            time_left=time_left2,
            to_visit=to_visit)

        time_left2 -= distances[position2][best_choice]+1
        total += max(0, time_left2*rates[best_choice])
        position2 = to_visit.pop(to_visit.index(best_choice))
        
    simulation_results.append(total)
    
print(max(simulation_results))
```

    2504


## Day 17


```python
def rock_generator(i, height):
    im = height+4
    rock_type = i%5
    if rock_type==0:
        rock = [3+k + im*1j for k in range(4)]
    elif rock_type==1:
        rock = [4 + (im+2)*1j] \
            + [3+k + (im+1)*1j for k in range(3)] \
            + [4 + im*1j]
    elif rock_type==2:
        rock = [5 + (im+2)*1j] \
            + [5 + (im+1)*1j] \
            + [3+k + im*1j for k in range(3)]
    elif rock_type==3:
        rock = [3 + (im+k)*1j for k in range(4)]
    elif rock_type==4:
        rock = [3+k + im*1j for k in range(2)] \
            + [3+k + (im+1)*1j for k in range(2)]

    return rock, rock_type


def wind_generator(j, input, rock):
    index = j%len(input)
    edges = [
        max(r.real for r in rock),
        min(r.real for r in rock)
        ]
    #current 
    if input[index]==-1 and 1 in edges:
        return 0, index
    elif input[index]==1 and 7 in edges:
        return 0, index
    return input[index], index


def offset_generator(pile):
    min_offset = [-1]*7
    for k in range(1, 8):
        for r in pile:
            if r.real==k:
                min_offset[k-1] = max(min_offset[k-1], r.imag)
    height = max(min_offset)
    min_offset = [height-h for h in min_offset]
    return tuple(min_offset)
```


```python
with open('data/day17.txt') as data:
    input = data.read() 

input = list(input)
input = [int(i.replace('>', '1').replace('<', '-1')) for i in input]
```


```python
height = 0
j = 0
pile = set()
seen_offsets = {}

i=0
additional=0
while i <= 1000000000000:
    # generate rock and rock_id
    rock, rock_id = rock_generator(i=i, height=height)
    if i==2022:
        print(height)
        skipping=True
    while True:
        # generate wind and wind_id
        wind, wind_id = wind_generator(
            j=j,
            input=input,
            rock=rock)
        # move wind index
        j+=1
        # see if wind can move rock without crashing into
        # the previous rocks, otherwie leave as is
        if len(pile.intersection(set([r+wind for r in rock])))==0:
            rock_ = set([r+wind for r in rock])
        else:
            rock_=rock
        # see if moving down one step crashed into previous rocks
        if len(pile.intersection(set([r-1j for r in rock_])))>0 or min(r.imag for r in rock_)==1:
            # add to rock pile
            for r in rock_: pile.add(r)
            # get max height
            height=max((r.imag for r in pile))
            # get contour of rocks from above
            offset = offset_generator(pile)
            # move rock index
            i+=1

            # see if current contour, rock_id and wind_id
            # has been seen before -> repeating pattern
            if (offset, rock_id, wind_id) in seen_offsets and skipping:
                # repeat as many times as possible with current limit
                old_i, old_height = seen_offsets[(offset, rock_id, wind_id)]
                skip = (1000000000000 - i) // (i - old_i)
                i += (i - old_i) * skip
                additional += skip * (height - old_height)
                seen_offsets = {}
            else:
                seen_offsets[offset, rock_id, wind_id]= (i, height)
            break
        # else move down one step
        else:
            rock = set([r-1j for r in rock_])

height+additional-1
```

    3137.0





    1564705882327.0



## Day 17


```python
input='''2,2,2
1,2,2
3,2,2
2,1,2
2,3,2
2,2,1
2,2,3
2,2,4
2,2,6
1,2,5
3,2,5
2,1,5
2,3,5'''

with open('data/day18.txt') as data:
    input = data.read() 

input = [list(map(int, line.split(','))) for line in input.splitlines()]
```


```python
class Cube:
    """
    Simple cube object with walls/edges
    """
    def __init__(
            self,
            position: list[int],
            ):
        self.position = tuple(position)
        self.edges = self._add_edges(position)
    

    def _add_edges(
            self,
            position
            ):
        edges = set()
        for i in range(len(position)):
            for j in [-0.5, 1]:
                position[i]+=j
                edges.add(tuple(position))
            position[i]-=0.5
        return edges
```


```python
cubes = []
for point in input:
    cubes.append(Cube(point))

edges = []
for cube in cubes:
    for edge in cube.edges:
        edges.append(edge)

values = {}
for edge in edges:
    if edge in values:
        values[edge]+=1
    else:
        values[edge]=1

# question 1
sum([value for value in values.values() if value==1])
```




    3610




```python
min_x, max_x = 10_000, -10_000
min_y, max_y = 10_000, -10_000
min_z, max_z = 10_000, -10_000

positions = set()

for cube in cubes:
    min_x = int(min(min_x, cube.position[0]))
    max_x = int(max(max_x, cube.position[0]))
    min_y = int(min(min_y, cube.position[1]))
    max_y = int(max(max_y, cube.position[1]))
    min_z = int(min(min_z, cube.position[2]))
    max_z = int(max(max_z, cube.position[2]))
    positions.add(cube.position)


# coords form a cube surrounding
# the shape given by the input 

min_x -= 1
max_x += 1
min_y -= 1
max_y += 1
min_z -= 1
max_z += 1


# get all reachable point in cube
# basically flood fill from min coordinates

water = set()
to_visit = [(min_x, min_y, min_z)]

while to_visit:
    current = to_visit.pop(0)
    water.add(current)
    
    for dir in [1, -1]:
        if min_x<=current[0]+dir<=max_x:
            x_tuple = (current[0]+dir, current[1], current[2])
            if x_tuple not in positions and x_tuple not in water:
                water.add(x_tuple)
                to_visit.append(x_tuple)
        
        if min_y<=current[1]+dir<=max_y:
            y_tuple = (current[0], current[1]+dir, current[2])
            if y_tuple not in positions and y_tuple not in water:
                water.add(y_tuple)
                to_visit.append(y_tuple)

        if min_z<=current[2]+dir<=max_z:
            z_tuple=(current[0], current[1], current[2]+dir)
            if z_tuple not in positions and z_tuple not in water:
                water.add(z_tuple)
                to_visit.append(z_tuple)


# all points NOT reached by above fill
# is either solid or trapped air

solid = []
for x in range(min_x, max_x+1):
    for y in range(min_y, max_y+1):
        for z in range(min_z, max_z+1):
            if (x, y, z) not in water:
                solid.append((x,y,z))


# apply approach used in q1 for all of 
# these points

cubes = []
for point in solid:
    cubes.append(Cube(list(point)))

edges = []
for cube in cubes:
    for edge in cube.edges:
        edges.append(edge)

values = {}
for edge in edges:
    if edge in values:
        values[edge]+=1
    else:
        values[edge]=1

# question 2
sum([value for value in values.values() if value==1])
```




    2082



## Day 19


```python
with open('data/day19.txt') as data:
    input = data.read() 

recipes = {}
for line in input.splitlines():
    words = line.split()
    recipe_n = int(words[1][:-1])
    recipes[recipe_n] = {}
    recipes[recipe_n]['r1_cost']= int(words[6])
    recipes[recipe_n]['r2_cost'] = int(words[12])
    recipes[recipe_n]['r3_cost1'] = int(words[18])
    recipes[recipe_n]['r3_cost2'] = int(words[21])
    recipes[recipe_n]['r4_cost1'] = int(words[27])
    recipes[recipe_n]['r4_cost2'] = int(words[30])
```


```python
def simulate(r1_cost, r2_cost, r3_cost1, r3_cost2, r4_cost1, r4_cost2, time):
    max_geodes = 0
    start = (0, 0, 0, 0, 1, 0, 0, 0, time)
    queue = deque([start])
    visited = set()
    while queue:
        ore, clay, obsidian, geode, r1, r2, r3, r4, t = queue.popleft()

        max_geodes = max(max_geodes, geode)
        if t==0:
            continue

        max_ore_use = max([r1_cost, r2_cost, r3_cost1, r4_cost1])
        # make sure we don't build more of r1 than necessary
        # if we cant use more than x ore per round, never
        # build more than x r1 
        if r1 >= max_ore_use:
            r1 = max_ore_use
        # same for r2
        if r2>=r3_cost2:
            r2 = r3_cost2
        # same for r3...
        if r3>=r4_cost2:
            r3 = r4_cost2

        # correct current amount of ore following above adjustments..
        if ore >= t*max_ore_use-r1*(t-1):
            ore = t*max_ore_use-r1*(t-1)
        # same for clay...
        if clay>=t*r3_cost2-r2*(t-1):
            clay = t*r3_cost2 - r2*(t-1)
        # same for obsidian...
        if obsidian>=t*r4_cost2-r3*(t-1):
            obsidian = t*r4_cost2-r3*(t-1)

        state = (
            ore,
            clay,
            obsidian,
            geode,
            r1, r2, r3, r4, t)

        # if state already seen, skip
        if state in visited:
            continue
        visited.add(state)

        # steps forward, either do nothing
        # or build a robot

        # do nothing
        queue.append((
            ore + r1,
            clay + r2,
            obsidian + r3,
            geode + r4,
            r1, r2, r3, r4, t-1))
        # build robot 1
        if ore >= r1_cost:
            queue.append((
                ore - r1_cost + r1,
                clay + r2,
                obsidian + r3,
                geode + r4,
                r1 + 1, r2, r3, r4, t-1))
        # build robot 2 etc..
        if ore >= r2_cost:
            queue.append((
                ore - r2_cost + r1,
                clay + r2,
                obsidian + r3,
                geode + r4,
                r1, r2 + 1, r3, r4, t-1))
        # build robot 3
        if ore >= r3_cost1 and clay >= r3_cost2:
            queue.append((
                ore - r3_cost1 + r1,
                clay - r3_cost2 + r2,
                obsidian + r3,
                geode + r4,
                r1, r2, r3+1, r4, t-1))
        # build robot 4
        if ore >= r4_cost1 and obsidian >= r4_cost2:
            queue.append((
                ore - r4_cost1 + r1,
                clay + r2,
                obsidian - r4_cost2 + r3,
                geode + r4,
                r1, r2, r3, r4+1, t-1))
    return max_geodes


question1 = []
question2 = []

for key in recipes.keys():
    max_geodes1 = simulate(
        r1_cost=recipes[key]['r1_cost'],
        r2_cost=recipes[key]['r2_cost'],
        r3_cost1=recipes[key]['r3_cost1'],
        r3_cost2=recipes[key]['r3_cost2'],
        r4_cost1=recipes[key]['r4_cost1'],
        r4_cost2=recipes[key]['r4_cost2'],
        time=24)
        
    question1.append(key * max_geodes1)

    if key<=3:
        max_geodes2 = simulate(
            r1_cost=recipes[key]['r1_cost'],
            r2_cost=recipes[key]['r2_cost'],
            r3_cost1=recipes[key]['r3_cost1'],
            r3_cost2=recipes[key]['r3_cost2'],
            r4_cost1=recipes[key]['r4_cost1'],
            r4_cost2=recipes[key]['r4_cost2'],
            time=32)
        question2.append(max_geodes2)

print(sum(question1))
print(question2[0]*question2[1]*question2[2])
```

    1681
    5394


## Day 20


```python
with open('data/day20.txt') as data:
   input = data.read()
input=[int(i) for i in input.splitlines()]

n = len(input)
input_=[input[i]+i*1j for i in range(len(input))]
to_do = deepcopy(input_)


while to_do:
    val = to_do.pop(0)
    idx = input_.index(val)
    j = (idx + val.real) % (n - 1)
    input_.pop(idx)
    input_.insert(int(j), val.real)

idx = input_.index(0)
nums = [1000, 2000, 3000]
# question 1
print(sum([input_[(idx + nums[i]) % n] for i in range(3)]))


input_=[input[i]*811589153+i*1j for i in range(len(input))]
to_do = deepcopy(input_)*10

while to_do:
    val = to_do.pop(0)
    idx = input_.index(val)
    j = (idx + val.real) % (n - 1)
    input_.pop(idx)
    input_.insert(int(j), val)

input_ = [i.real for i in input_]

idx = input_.index(0)
nums = [1000, 2000, 3000]
# question 2
print(sum([input_[(idx + nums[i]) % n].real for i in range(3)]))
```

    3466.0
    9995532008348.0


## Day 21


```python
with open('data/day21.txt') as data:
        input = data.read() 
```


```python
values = {k.split(': ')[0]:{'expression':k.split(': ')[1]} for k in input.splitlines()}
to_do = []

for key in values.keys():
    # check if value already number
    try:
        values[key]['expression']=float(values[key]['expression'])
        values[key]['done']=True
    # otherwise get expression and needed numbers
    except:
        values[key]['needs'] = values[key]['expression'].split()[0::2]
        values[key]['done']=False
        to_do.append(key)

    if key=='root':
        key1 = values[key]['needs'][0]
        key2 = values[key]['needs'][1]
```


```python
to_do_ = deepcopy(to_do)
values_ = deepcopy(values)

while to_do_:
    current = to_do_.pop(0)
    expression = False
    needs = values_[current]['needs']
    
    if all([values_[n]['done'] for n in needs]):
        expression = (str(values_[needs[0]]['expression'])
            + values_[current]['expression'].split()[1]
            + str(values_[needs[1]]['expression']))
    
    if expression:
        values_[current]['expression'] = eval(expression)
        values_[current]['done']=True
    else:
        to_do_.append(current)

# question  1    
values_['root']['expression']
```




    66174565793494.0




```python
diff = []
i=1

while True:
    i+=1
    to_do_ = deepcopy(to_do)
    values_ = deepcopy(values)
    values_['humn']['expression']=i
    
    while to_do_:
        current = to_do_.pop(0)
        expression = False
        needs = values_[current]['needs']
        
        if all([values_[n]['done'] for n in needs]):
            expression = (str(values_[needs[0]]['expression'])
                + values_[current]['expression'].split()[1]
                + str(values_[needs[1]]['expression']))
        
        if expression:
            values_[current]['expression'] = eval(expression)
            values_[current]['done']=True
        else:
            to_do_.append(current)
    
    if values_[key1]['expression']==values_[key2]['expression']:
        break

    diff.append(values_[key1]['expression']-values_[key2]['expression'])

    # ghetto gradient descent
    if len(diff)>1:
        slope = diff[0]-diff[1]
        to_increase = diff[1]/slope
        diff.pop(0)
        if to_increase>1:
            i+=int(to_increase//1.05)

# question 2
print(i)
```

    3327575724809

