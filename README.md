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
        end=None
        ):
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
                elif neighbor==end:
                    return D
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
D = dijkstra(m, start, end=end)
D[end]
```




    339




```python
m = Mountain(matrix=input, inverse=True)
D = dijkstra(m, end)
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

