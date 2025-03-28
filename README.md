### 이 저장소(Repository)는 「Pytorch 기반 GPT 모델 및 모듈 라이브러리」에 대한 내용을 다루고 있습니다.

***
작성자: YAGI<br>

최종 수정일: 2025-03-28
+ 2025.03.27: 코드 작성 완료
+ 2025.03.28: READ ME 작성 완료
***

<br>

***
+ 프로젝트 기간: 2025-03-22 ~ 2025-03-28
***
<br>

## 프로젝트 내용
&nbsp;&nbsp; 본 프로젝트에서는 Pytorch 기반의 다양한 GPT 모델과 `RoPE(Rotary PositionalEmbedding)`, `MoE(Mixture of Expert)`와 같은 GPT 모델에 사용되는 여러 모듈을 제공한다. 나아가 Hugging Face에서 제공하는 `databricks/databricks-dolly-15k` 데이터셋을 이용하여 몇 가지 모델의 평가 지표를 제시한다. Table 1는 `gptModules` 라이브러리에서 제공하는 GPT 모델에 대한 설명이다.

<br>

|Model|Description|Code|
|:---:|:---|:---:|
|GPT-1|-|`models.GPT(...)`|
|GPT-2|*· Pre-Norm Layer*|`models.GPT2(...)`|
|ALiBi GPT|*· Pre-Norm Layer <br> · ALiBi Embedding Layer*|`models.ALiBiGPT(...)`|
|LLaMA|*· Pre-Norm Layer <br> · RoPE Embedding Layer <br> · Group Query Attention <br> · RMS Normalization*|`models.LLaMA(...)`|
|DeepSeek V2|*· Pre-Norm Layer <br> · RoPE Embedding Layer <br> · Multi Head Latent Attention <br> · DeepSeek MoE <br> · RMS Normalization*|`models.DeepSeek(...)`|

<br> 

<b>Table 1</b>. gptModules Library Models.

<br>

&nbsp;&nbsp; `gptModules` 라이브러리의 `layers`를 사용하여 모델뿐만 아니라 GPT에 사용되는 다양한 모듈에 접근할 수 있다. Table 2는 본 라이브러리에서 제공하는 GPT 모듈에 대한 설명이다.

<br>


<style type="text/css">
.tg  {border-collapse:collapse;border-color:#93a1a1;border-spacing:0;}
.tg td{background-color:#fdf6e3;border-color:#93a1a1;border-style:solid;border-width:1px;color:#002b36;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#657b83;border-color:#93a1a1;border-style:solid;border-width:1px;color:#fdf6e3;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-dxyq{border-color:inherit;font-size:x-small;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-hj9z{border-color:#656565;font-size:x-small;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-c7c9{border-color:#656565;font-size:x-small;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-cjj3{background-color:#eee8d5;font-size:x-small;font-style:italic;text-align:left;vertical-align:top}
.tg .tg-1l0q{font-size:x-small;text-align:center;vertical-align:top}
.tg .tg-ue70{background-color:#eee8d5;border-color:inherit;font-size:x-small;font-weight:bold;text-align:center;
  vertical-align:middle}
.tg .tg-ejl1{font-size:x-small;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-hquy{background-color:#eee8d5;border-color:inherit;font-size:x-small;font-style:italic;text-align:left;vertical-align:top}
.tg .tg-mg0j{background-color:#eee8d5;border-color:inherit;color:#333333;font-size:x-small;text-align:center;vertical-align:top}
.tg .tg-it0k{border-color:inherit;font-size:x-small;font-style:italic;text-align:left;vertical-align:top}
.tg .tg-0ocd{border-color:inherit;color:#333333;font-size:x-small;text-align:center;vertical-align:top}
.tg .tg-i3ef{font-size:x-small;font-style:italic;text-align:left;vertical-align:top}
.tg .tg-96f4{background-color:#eee8d5;font-size:x-small;text-align:center;vertical-align:top}
</style>
<table class="tg" style="undefined;table-layout: fixed; width: 667px"><colgroup>
<col style="width: 114.090909px">
<col style="width: 275.181818px">
<col style="width: 278.181818px">
</colgroup>
<thead>
  <tr>
    <th class="tg-hj9z">Layer</th>
    <th class="tg-c7c9">Module</th>
    <th class="tg-ejl1">Code</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-ue70" rowspan="4">Embedding</td>
    <td class="tg-hquy">· Embedding</td>
    <td class="tg-mg0j">layers.Embeddings(...)</td>
  </tr>
  <tr>
    <td class="tg-it0k">· Embedding Without Positional Embedding</td>
    <td class="tg-0ocd">layers.EmbeddingsWithoutPosition(...)</td>
  </tr>
  <tr>
    <td class="tg-hquy">· Rotary Positional Embedding</td>
    <td class="tg-mg0j">layers.RotaryPositionalEmbeddings(...)</td>
  </tr>
  <tr>
    <td class="tg-it0k">· ALiBi(Attention with Linear Biases) Positional Embedding</td>
    <td class="tg-0ocd">layers.ALiBiEmbeddings(...)</td>
  </tr>
  <tr>
    <td class="tg-ue70">Normalization</td>
    <td class="tg-hquy">· RMS Normalization</td>
    <td class="tg-mg0j">layers.RMSNorm(...)</td>
  </tr>
  <tr>
    <td class="tg-dxyq" rowspan="6">Multi Head Attention</td>
    <td class="tg-it0k">· Masked Multi Head Attention</td>
    <td class="tg-0ocd">layers.MaskedMultiHeadAttention(...)</td>
  </tr>
  <tr>
    <td class="tg-hquy">· ALiBi Attention</td>
    <td class="tg-mg0j">layers.ALiBiAttention(...)</td>
  </tr>
  <tr>
    <td class="tg-it0k">· GQA(Grouped Query Attention) **with RoPE</td>
    <td class="tg-0ocd">layers.GroupedQueryAttention(...)</td>
  </tr>
  <tr>
    <td class="tg-hquy">· GQA Without RoPE</td>
    <td class="tg-mg0j">layers.GroupedQueryAttentionWithoutRoPE(...)</td>
  </tr>
  <tr>
    <td class="tg-it0k">· Multi Head Latent Attention **with RoPE</td>
    <td class="tg-0ocd">layers.MultiHeadLatentAttention(...)</td>
  </tr>
  <tr>
    <td class="tg-hquy">· Multi Head Latent Attention Without RoPE</td>
    <td class="tg-mg0j">layers.MultiHeadLatentAttentionWithoutRoPE(...)</td>
  </tr>
  <tr>
    <td class="tg-dxyq" rowspan="2">Feed Forward</td>
    <td class="tg-it0k">· Position Wise Feed Forward</td>
    <td class="tg-0ocd">layers.PositionWiseFeedForward(...)</td>
  </tr>
  <tr>
    <td class="tg-hquy">· Deep Seek V2 Mixture of Expert(MoE)</td>
    <td class="tg-mg0j">layers.DeepSeekMoE(...)</td>
  </tr>
  <tr>
    <td class="tg-dxyq" rowspan="6">Transfomer Block</td>
    <td class="tg-it0k">· Transformer Block</td>
    <td class="tg-0ocd">layers.TransformerBlock(...)</td>
  </tr>
  <tr>
    <td class="tg-hquy">· Pre-Norm Transformer Block</td>
    <td class="tg-mg0j">layers.PreNormTransformerBlock(...)</td>
  </tr>
  <tr>
    <td class="tg-it0k">· ALiBi Transformer Block</td>
    <td class="tg-0ocd">layers.ALiBiTransformerBlock(...)</td>
  </tr>
  <tr>
    <td class="tg-hquy">· Grouped Query Transformer Block</td>
    <td class="tg-mg0j">layers.GroupedQueryTransformerBlock(...)</td>
  </tr>
  <tr>
    <td class="tg-i3ef">· Deep Seek Transformer Block</td>
    <td class="tg-1l0q">layers.DeepseekTransformerBlock(...)</td>
  </tr>
  <tr>
    <td class="tg-cjj3">· Deep Seek Transformer Block Without RoPE</td>
    <td class="tg-96f4">layers.DeepSeekTransformerBlockWithoutRoPE(...)</td>
  </tr>
</tbody></table>


<b>Table 2</b>. gptModules Library Layers.

<br>


<br>

&nbsp;&nbsp; Hugging Face에서 제공하는 `databricks/databricks-dolly-15k` 데이터셋을 이용하여 `gputModules`의 각 모델을 학습하였다. Tokenizer는 `GPT2Tokenizer`를 사용하였다. `n_layer=9`, `n_heads=10`, `d_model=560`, `d_ff=2304`를 모든 모델의 기본적인 아키텍처 하이퍼파라미터로 설정하였다. 별개의 하이퍼파라미터를 요구하는 모델인 `LLaMA`의 경우 `n_groups=5`, `rope_base=500_000`으로 설정하였으며, `DeepSeek`의 경우 `n_shared=1`, `d_ff=576`, `top_k=2`, `d_kv_comp=12`, `d_rope=14`, `rope_base=10_000`으로 설정하였다. 옵티마이저로 `AdamW`를 사용하였다. learning rate의 경우 초기 0에서 최대 0.00022까지 상승하여 이후 점차 감소하도록 learning rate decay를 수행하였다. 총 Epoch은 8회이며 Batch 사이즈는 8이다. Fig 1은 Train set과 Validation set에 대한 각 모델의 학습에 따른 Loss의 변화를 제시한 것으로, 최종 Test set에 대한 Loss 및 학습 수행 과정 간의 Iter / Sec를 Table 3을 통해 확인할 수 있다.

<br><img src='./figures/GPT_8.png' height=250><img src='./figures/GPT2_8.png' height=250>
<br><img src='./figures/ALiBiGPT_8.png' height=250><img src='./figures/LLaMA_8.png' height=250>
<br><img src='./figures/DeepSeek_8.png' height=250>

<br>

<b>Fig 1</b>. Loss Graph of Models about Train and Validation Dataset.

<br>

|Model|Test Loss|Iter/Sec|
|:---:|:---|:---:|
|GPT-1|2.402|<span style="color:red">**8.262**</span>|
|GPT-2|3.210|7.235|
|ALiBi GPT|<span style="color:red">**1.702**</span>|7.132|
|LLaMA|2.681|4.523|
|DeepSeek V2|2.189|2.688|

<br>

<b>Table 3</b>. Loss and *Iter/sec* of Test Dataset.


<br><br>

## Getting Start

### Example
```python
#XOR Example
$ python main.py --mode logic --device cuda

#MNIST Example
$ python main.py --mode mnist --device cuda

#CIFAR-10 Example
$ python main.py --mode cifar10 --depth 2 --device cuda

#Auto Encoder Example
$ python main.py --mode AE --depth 3 -- device cuda

#학습 완료 후 './figures/' 디렉토리에 그래프가 저장됨.

```
<br>

### Use Picky Activation
```python
import torch
import activation


x = torch.randn(size=(1, 5)) #Input Tensor: Batch(1) x Feature(5)
print(f'input: {input_tensor}')


#함수형 활성화 함수
y_hat = activation.picky_(x)
print(f'functional y_hat: {y_hat}')


#클래스형 활성화 함수
actF = activation.Picky()
y_hat = actF(x)
print(f'functional y_hat: {y_hat}')

```
***

<br><br>

## 개발 환경
**Language**

    + Python 3.9.12

    
**Library**

    + tqdm 4.64.1
    + pytorch 1.12.0