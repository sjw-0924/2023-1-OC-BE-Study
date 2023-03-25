---
jupyter:
  kernelspec:
    display_name: Python 3.8.16 (\'tensorflow\')
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.2
  nbformat: 4
  nbformat_minor: 2
  orig_nbformat: 4
  vscode:
    interpreter:
      hash: 0a52b89d3fc945414f6f8695388390711154e06ab9d9a15bbc22f76a3173a686
---

::: {.cell .markdown}
# Visualization

[]{.image .placeholder original-image-src="./main.jpg"
original-image-title=""}

우리는 지난 시간 데이터를 읽어오고 원하는 형식으로 가공하게 도와주는
`pandas`에 대해 공부해봤습니다. 하지만 누군가에게 내가 데이터에서 얻어낸
인사이트를 설명할 때, 말로 구구절절 설명하는 게 정말 최선일까요? 처음
보는 사람에게는 어지러울 수 있는 표를 그냥 무작정 보여준다는 것은,
어떻게 보면 에너지를 낭비할 수 있는 방법이기도 합니다.

#### **백문이 불허일견**!

때로는 100줄짜리 글보다 그림 하나가 더 효과적인 설명이 되기도 합니다.
실제로 얻어낸 데이터로 AI나 ML을 돌려보기 전에, 데이터에 대해 미리
시각화를 진행함으로써 어떤 방향으로 분석을 진행해야겠구나\~ 하며 전략을
수립하는 것이 보통입니다. 지피지기면 백전백승이라고, 시각화는 모델링에
있어서도 `적을 아는 방법`이라고 할 수 있습니다.

이번 챕터에서 우리는 `matplotlib`과 `seaborn`으로 데이터 시각화를
진행합니다. 한번 즐겨봐요 :)

`<br>`{=html}

------------------------------------------------------------------------

**알림**

-   본 컨텐츠는 강의 형식이 아닌, 스스로 공부하시는 분들을 위한 일종의
    문제집 입니다.
-   데이터라는 큰 바다에서 여러분이 쓸 데 없는 시간 낭비 없이 바로바로
    핵심을 배우실 수 있도록 커리큘럼을 짜봤습니다.
-   이 컨텐츠의 문제들만 해결한다고 실력이 오르지 않습니다. 본 컨텐츠의
    목적은 문제를 해결하는 과정에서 발생하는 고민과 질문을 통한 실력
    향상입니다.
-   문제에서 절대 떠먹여주지 않습니다. 물고기를 잡아주는 것이 아닌,
    물고기를 잡는 방법을 여러분이 이 컨텐츠를 통해 알아가셨으면 합니다.
:::

::: {.cell .markdown}
# 1. 시각화 기본 {#1-시각화-기본}

-   \"어떤 일이 쉬워보이는 건, 그 사람이 진짜 잘하기 때문이다.\"라는
    말이 있죠? 그렇게 잘 하기 위해서는 뭐든 기본기가 중요한 것같습니다.
    시각화의 가장 기본인 plot들을 그려보면서, `matplotlib`과 `seaborn`의
    기본적인 사용법에 익숙해져봅시다!
:::

::: {.cell .markdown}
❗ 잠깐 ❗\
들어가기 전에, `matplotlib`과 `seaborn`의 차이가 뭘까요?\
간단하게만 말씀 드리면, `seaborn`이 `matplotlib`의 간편화된 버전이라고
보시면 될 것같습니다. `seaborn`은 통계적인 부분의 시각화를 위해
탄생했고, 때문에 더 깔끔하고 직관적인 시각화를 지원합니다. 다만 이
`seaborn`도 결국 `matplotlib`기반이기 때문에 둘 다 사용하는 방법을
익히신 다음, 시각화 하는 목적에 따라 선택하시는 것이 좋겠습니다.

-   요약 : 빨리 간단하게 그릴 거면 `seaborn` / 세심하게 그리고 싶으면
    `matplotlib`
:::

::: {.cell .markdown}
### 1-1 line plot {#1-1-line-plot}

**문제**\
아래의 x와 y값을 이용하여 line plot을 그려보세요.

``` python
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
y = [123, 1, 59, 40, 10, 100]

> CODE HERE !!

plt.show()
```

검색 힌트 : matplotlib line plot

*예시*\
[]{.image .placeholder original-image-src="./1-1answer.png"
original-image-title=""}
:::

::: {.cell .code execution_count="4"}
``` python
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
y = [123, 1, 59, 40, 10, 100]

plt.plot(x,y)
plt.show()
```

::: {.output .display_data}
![](vertopal_3c21644830434ee7913d77e0d3aa7c8a/e52ba49f0f9f3c5b11687a364e0587e8f1fca4f5.png)
:::
:::

::: {.cell .markdown}
### 1-2 Title and Bar plot {#1-2-title-and-bar-plot}

**문제**\
아래의 x와 y값을 이용하여 bar plot을 그려보세요.\
단, 제목은 `Shape Like W`라고 붙여주세요.

``` python
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
y = [123, 1, 59, 40, 10, 100]

> CODE HERE !!

plt.show()
```

검색 힌트 : matplotlib bar plot, matplotlib title

*예시*\
[]{.image .placeholder original-image-src="./1-2answer.png"
original-image-title=""}
:::

::: {.cell .code execution_count="6"}
``` python
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
y = [123, 1, 59, 40, 10, 100]

plt.bar(x,y)
plt.title("Shape Like W")

plt.show()
```

::: {.output .display_data}
![](vertopal_3c21644830434ee7913d77e0d3aa7c8a/be494fef97053cc23f8747ebda2935be34ee30b4.png)
:::
:::

::: {.cell .markdown}
### 1-3 Scatter plot {#1-3-scatter-plot}

**문제**\
`seaborn`에서 제공하는 penguin 데이터셋을 가지고 (지난번엔 다운받았죠?
이번엔 내장 데이터셋을 활용합니다) **몸무게와 부리의 길이 관계를
알아보고 싶습니다.**\
scatter plot을 활용하여 두 변수간의 관계를 scatter plot으로 시각화 하는
코드를 작성해주세요.

단, 제목은 \"Body Mass vs Bill Length\"로 해주세요.\
(X축에 몸무게, Y축에 부리의 길이를 표현해주세요.)

``` python
import matplotlib.pyplot as plt
import seaborn as sns

penguin_df = sns.load_dataset("penguins")

> CODE HERE !!

plt.show()
```

검색 힌트 : matplotlib scatter plot

*예시*\
[]{.image .placeholder original-image-src="./1-3answer.png"
original-image-title=""}
:::

::: {.cell .code execution_count="18"}
``` python
import matplotlib.pyplot as plt
import seaborn as sns

penguin_df = sns.load_dataset("penguins")

plt.scatter(penguin_df["body_mass_g"],penguin_df["bill_length_mm"])
plt.title("Body Mass vs Bill Length")
plt.show()
```

::: {.output .display_data}
![](vertopal_3c21644830434ee7913d77e0d3aa7c8a/6dd8b88531039f0cb0a56511e81f56782d0e068e.png)
:::
:::

::: {.cell .markdown}
### 1-4 subplots and figsize {#1-4-subplots-and-figsize}

**문제**\
주어진 X와 Y를 가지고, 위에서 배운 line plot, bar plot, scatter plot을
한번에 그려주세요. 단, 아래의 조건을 따른 코드를 작성해주세요.

1.  figsize를 12 x 5로 지정합니다.
2.  가로로 3개의 그래프를 한 줄에 그려주세요.
3.  line plot, bar plot, scatter plot 순서대로 그려주세요.

``` python
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
y = [123, 1, 59, 40, 10, 100]

> CODE HERE !!

plt.show()
```

검색 힌트 : matplotlib subplots, matplotlib 여러개 그리기, matplotlib
figsize

*예시*\
[]{.image .placeholder original-image-src="./1-4answer.png"
original-image-title=""}
:::

::: {.cell .code execution_count="26"}
``` python
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
y = [123, 1, 59, 40, 10, 100]


plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.plot(x,y)


plt.subplot(1,3,2)
plt.bar(x,y)


plt.subplot(1,3,3)
plt.scatter(x,y)
plt.show()
```

::: {.output .display_data}
![](vertopal_3c21644830434ee7913d77e0d3aa7c8a/5531840b66f8d2e3d3be73a3d396d0527b1524c5.png)
:::
:::

::: {.cell .markdown}
### 1-5 생각해보기 {#1-5-생각해보기}

> 1.  우리는 지금까지 line plot, bar plot, scatter plot을 그려봤습니다.
>     어떤 데이터를 다루냐, 시각화 하느냐에 따라 최적의 plot이 있겠죠?
>     몸무게와 키의 상관관계를 그려보고 싶은데 bar plot보다는 scatter
>     plot이 적절할 것이며, 주가를 표시할 땐 scatter plot보다는 line
>     plot이 적절할 것입니다. 그 외 각 plot이 어떨 때 유용할지 한번
>     생각해봅시다

-   생각 :

> 1.  그런데, 지금까지 그렸던 그림들을 보니, 어떤 그림은 `plt`로 그리고,
>     어떤 건 `fig, axes`로 그렸습니다. 둘은 어떤 차이가 있을까요? 한번
>     검색을 통해 알아본 후(ChatGPT도 활용해보세요!) 여러분이 알아낸
>     내용을 정리해보세요

-   plt vs ax :
:::

::: {.cell .markdown}
### 1-6 Dittribution {#1-6-dittribution}

**문제**\
아래의 코드를 활용하여, 펭귄 몸무게의 분포를 `seaborn`으로 그리는 코드를
작성해주세요.\
단, 색은 `orange`로 하고, density line 밑을 채워주세요.

``` python
import matplotlib.pyplot as plt
import seaborn as sns

penguin_df = sns.load_dataset("penguins")

> CODE HERE !!

plt.show()
```

검색 힌트 : seaborn kdeplot, seaborn color

*예시*\
[]{.image .placeholder original-image-src="./1-6answer.png"
original-image-title=""}
:::

::: {.cell .code execution_count="33"}
``` python
import matplotlib.pyplot as plt
import seaborn as sns

penguin_df = sns.load_dataset("penguins")

sns.kdeplot(penguin_df["body_mass_g"],fill=True, color="orange")

plt.show()
```

::: {.output .display_data}
![](vertopal_3c21644830434ee7913d77e0d3aa7c8a/aa389ec66b7114b35986c6cf29e7f997605ed4cf.png)
:::
:::

::: {.cell .markdown}
### 1-7 Pie Chart {#1-7-pie-chart}

**문제**\
주어진 데이터로 다음과 같은 파이차트를 그리는 코드를 작성해주세요.\
단, FORD의 expode는 `0.2`입니다.

``` python
import matplotlib.pyplot as plt
cars=['AUDI', 'BMW', 'FORD', 'TESLA', 'JAGUAR']
data=[23, 29, 41, 12, 17]

> CODE HERE !!

plt.show()
```

`<br>`{=html}

👀추가 학습👀

-   색이 그렇게 이쁜 편은 아니죠? `matplotlib`에서 활용할 수 있는 색은
    무엇인지, 투명도는 어떻게 설정할 수 있을지 알아봅시다! -\> `cmap`을
    한번 검색해보세요..!

검색 힌트 : matplotlib pie chart, plt shadow, plt explode, plt autopct

*예시*\
[]{.image .placeholder original-image-src="./1-7answer.png"
original-image-title=""}
:::

::: {.cell .code execution_count="39"}
``` python
import matplotlib.pyplot as plt
cars=['AUDI', 'BMW', 'FORD', 'TESLA', 'JAGUAR']
data=[23, 29, 41, 12, 17]
explode=[0,0,0.2,0,0]
plt.pie(data,labels=cars,autopct="%.1f%%",explode=explode,counterclock=False,shadow=True)

plt.show()
```

::: {.output .display_data}
![](vertopal_3c21644830434ee7913d77e0d3aa7c8a/da3f85ffbbe801a44a92afe514d4c817cd3b8531.png)
:::
:::

::: {.cell .markdown}
### 1-8 Box plot {#1-8-box-plot}

**문제**\
주어진 데이터셋에서 파티의 size별로 tip의 분포가 얼마나 차이가 나는지
**\"이상치와 함께\"** 확인할 수 있는 코드를 작성해주세요.\
그리고 해당 시각화가 어떤 의미를 가지는지 설명해주세요.

``` python
import matplotlib.pyplot as plt
import seaborn as sns
tips=sns.load_dataset('tips')

> CODE HERE !!

plt.show()
```

`<br>`{=html}

👀추가 학습👀

-   이상치란 무엇인가요? 이상치를 확인할 수 있는 시각화 기법은 2가지가
    있습니다(검색 힌트를 잘 보세요!) 각각에 대해 한번 검색해보고, 어디에
    사용될 수 있을지 생각해봅시다.

검색 힌트 : seaborn boxplot, seaborn violin plot

*예시*\
[]{.image .placeholder original-image-src="./1-8answer.png"
original-image-title=""}
:::

::: {.cell .code execution_count="6"}
``` python
import matplotlib.pyplot as plt
import seaborn as sns
tips=sns.load_dataset('tips')
sns.boxplot(y='tip',x='size',data=tips)

plt.show()
```

::: {.output .display_data}
![](vertopal_3c21644830434ee7913d77e0d3aa7c8a/df8b1bb1893bc3105a989bb70e4fb7942217fd2b.png)
:::
:::

::: {.cell .markdown}
# 2.Decorating {#2decorating}

-   기본적인 그림을 그렸습니다. 근데 파란색으로만 그림을 그리니 그림이
    많이 이쁘지 않습니다. 2부에선 그림을 좀 더 예쁘게 그리는 방법에 대해
    알아보겠습니다.
-   여기부턴 객체지향으로 시각화를 하시는 게 더 좋을 겁니다! 우린 이제
    그림의 요소 하나하나를 건드려야 하거든요 ㅎㅎ
:::

::: {.cell .markdown}
### 2-1 Xlabel, Ylabel and Colors {#2-1-xlabel-ylabel-and-colors}

**문제**\
주어진 데이터를 이용하여 *예시* 와 같은 그림을 그리는 코드를
작성해주세요.

1.  X축과 Y축에 이름이 있습니다.
    -   폰트 크기는 14, Bold로, 색은 회색으로 해주세요.
    -   X, Y 축과의 거리는 12만큼 띄워줍니다.
2.  그래프 자체의 색깔과 경계선의 색깔이 존재합니다.
    -   색은 \"lightgray\", 경계선은 \"gray\"로 지정해주세요.

``` python
import matplotlib.pyplot as plt

X = ["Tom", "Amy", "Jim", "Bill", "Pegi"]
Y = [8, 23, 52, 37, 64]

> CODE HERE !!

plt.show()
```

`<br>`{=html}

검색 힌트 : matplotlib axis label, ax.set_xlabel, ax.set_ylabel,
matplotlib bar color, matplotlib bar edgecolor, matplotlib label
fontdict, matplotlib label fontsize, matplotlib label fontweight,
matplotlib label color, matplotlib tight_layout

*예시*\
[]{.image .placeholder original-image-src="./2-1answer.png"
original-image-title=""}
:::

::: {.cell .code execution_count="17"}
``` python
import matplotlib.pyplot as plt

X = ["Tom", "Amy", "Jim", "Bill", "Pegi"]
Y = [8, 23, 52, 37, 64]
plt.bar(X,Y,color='lightgray',edgecolor='gray')
plt.xlabel("Salesperson",labelpad=12,fontdict={'color':'gray','size':'14','weight':'bold'})
plt.ylabel("Revenue (1000$)",labelpad=12,fontdict={'color':'gray','size':'14','weight':'bold'})

plt.show()
```

::: {.output .display_data}
![](vertopal_3c21644830434ee7913d77e0d3aa7c8a/952eb14b63fa7eb3fc5f7df76a7bd704ec766e65.png)
:::
:::

::: {.cell .markdown}
### 2-2 Text Annotation, 수직선과 수평선 {#2-2-text-annotation-수직선과-수평선}

**문제**\
주어진 데이터를 이용하여 *예시* 와 같은 그림을 그리는 코드를
작성해주세요.

1.  각 bar 위에 annotation이 존재합니다.
    -   검정색으로, center에 정렬해주세요. y값보다 4 낮은 위치해 있어야
        합니다.(힌트 : parameter를 잘 찾아보세요..!)
2.  y=45 지점에 파란 선을 넣고, y=15 지점에 빨간 선을 그려주세요.
    -   두 선은 모두 dash style로 그려야 합니다.
    -   단, 빨간 선의 밑엔 연한 빨강으로 색칠이 돼야 합니다. (힌트 :
        색을 연하게 하는 parameter를 찾아보세요!)

``` python
import matplotlib.pyplot as plt

X = ["Tom", "Amy", "Jim", "Bill", "Pegi"]
Y = [8, 23, 52, 37, 64]

> 2-1 정답 코드

> CODE HERE !!

plt.show()
```

`<br>`{=html}

검색 힌트 : matplotlib 객체지향 텍스트 추가, matplotlib 객체지향 텍스트
색상 변경,matplotlib axes 수평선, matplotlib axes 수직선, matplotlib
axes 음영, matpotlib zorder

*예시*\
[]{.image .placeholder original-image-src="./2-2answer.png"
original-image-title=""}
:::

::: {.cell .code execution_count="55"}
``` python
import matplotlib.pyplot as plt

X = ["Tom", "Amy", "Jim", "Bill", "Pegi"]
Y = [8, 23, 52, 37, 64]

plt.bar(X,Y,color='lightgray',edgecolor='gray',zorder=2)
plt.xlabel("Salesperson",labelpad=12,fontdict={'color':'gray','size':'14','weight':'bold'})
plt.ylabel("Revenue (1000$)",labelpad=12,fontdict={'color':'gray','size':'14','weight':'bold'})

for i in range(len(X)):
    plt.text(i, Y[i]-4, Y[i], ha='center', color='black')
    
plt.axhline(45,color='blue',linestyle='--',zorder=1)
plt.axhline(15,color='red',linestyle='--',zorder=1)

plt.axhspan(0, 15, color='red', alpha=0.2, zorder=1)

plt.show()
```

::: {.output .display_data}
![](vertopal_3c21644830434ee7913d77e0d3aa7c8a/40e1fd312db605a721ff6b9fabf46de138998b31.png)
:::
:::

::: {.cell .markdown}
#### 🚀이것도 해볼래요?🚀

-   딥러닝이나 머신러닝을 하다보면 모델의 예측 결과나 이미지들을 한 눈에
    보기 쉽게 그려야하는 경우가 있습니다. 바로 아래처럼요!

    > []{.image .placeholder original-image-src="./exam1.png"
    > original-image-title=""} []{.image .placeholder
    > original-image-src="./exam2.png" original-image-title=""}

`<br>`{=html}

-   위와 같이 그리려면, 이미지와 결과 하나하나를 위에서 했던 것처럼 입력
    해야할까요? axes 하나하나마다 그래프의 결과를 수작업으로 집어넣는
    건\... 생각만 해도 너무 끔찍합니다.
-   그렇게 하기 싫어서 우리가 코딩을 배운 거죠! for문으로 여러개의
    그림을 한번에 그리는 방법이 있습니다.
-   `matplotlib for문으로 여러개 그리기`를 검색해보시면 `plt.subplot`을
    이용하는 방법과 `plt.subplots`, `ravel()` 함수 등을 이용하는
    방법들이 있습니다.
-   **여러분들이 검색을 통해 공부한 내용을 아래에 정리해보세요!** 지금
    당장은 \"하 쒸 이걸 왜 해..\" 하시겠지만, 모델링 결과를 시각화 할 때
    분명 검색해보실 겁니다 ㅎㅎ

> 정리한 내용
:::

::: {.cell .code}
``` python
# 그래프의 갯수 만큼 n*m 행렬을 생성해서 subplots 으로 만들어준다. subplot(nrow=n,ncol=m, figsize(x,y))''' for row in range(n):
'''for row in range(n):     
     for col in range(m):
         index=row*m+col
으로 index를 설정 후 for 문으로 한 개 씩 그래프를 그린다. 
 '''        
```
:::

::: {.cell .markdown}
# 2주차 종료 {#2주차-종료}

-   시각화 문제를 모두 푸신 여러분!! 너무 고생 많으셨습니다 ㅎㅎ 위에서
    배웠던 내용들을 잘 활용하면, 여러분은 굳이 엑셀을 활용하지 않으셔도
    대용량의 데이터를 파이썬으로 금방 시각화하실 수 있게 되었습니다.
-   시각화는 많은 연습이 필요합니다. 어떻게 하면 데이터가 말하고 있는
    정보를 잘 전달할 수 있을지, 어떻게 그려야 직관적으로 와닿게 그릴 수
    있는지 고민해야 합니다.
-   그래도 여기까지 오신 분들 너무 고생많으셨습니다. 중간에 OOP로 그리는
    방식이 익숙하지 않으실 수 있는데, 많이 고민하신 만큼, 여러분의
    실력은 올랐을 겁니다.

## 숙제

-   안타깝게도 숙제가 있습니다.. 껄껄
-   [이곳](https://www.kaggle.com/c/titanic)에서 데이터를 받아 노트북이
    존재하는 폴더에 넣어주세요!
-   해당 데이터는 Titanic 승객들의 생존 여부와 그들의 특징을 담아낸
    데이터입니다.
-   해당 데이터로 우리가 지금까지 배웠던 시각화 기법들을 통해 **여러분
    나름대로의 정보를 시각화해보세요.**
-   어느 방식이든 상관 없습니다. 칼럼 두 개 관의 관계를 보기 위해
    `scatter plot`을 보셔도 되고, 탑승객의 성비를 확인해보기 위해
    `bar plot` 혹은 `seaborn`의 `countplot`을 활용할 수도 있습니다.
-   정답은 없습니다! 여러분들만의 시각화 결과를 \"블로그\"에 업로드
    해주세요!
:::

::: {.cell .code execution_count="110"}
``` python
import matplotlib.pyplot as plt
import seaborn as sns


titanic_df=sns.load_dataset("titanic")
plt.figure(figsize=(12,5))
plt.subplot(1,3,1) 

survived_by_class = titanic_df.groupby(['class'])['survived'].sum()
sns.barplot(x=survived_by_class.index,y=survived_by_class.values)
plt.ylabel("The number of survived people")

plt.subplot(1,3,2)

fare_by_class=titanic_df.groupby(['class'])['fare'].mean()
sns.barplot(x=fare_by_class.index,y=fare_by_class)
plt.ylabel("fare")

plt.subplot(1,3,3)

survive_by_parch=titanic_df.groupby(['parch'])['survived'].sum()
sns.barplot(x=survive_by_parch.index, y=survive_by_parch)
plt.show()
```

::: {.output .display_data}
![](vertopal_3c21644830434ee7913d77e0d3aa7c8a/455299278346cf8a75e633ade45e3e825bf878c8.png)
:::
:::
