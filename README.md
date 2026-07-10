---
license: creativeml-openrail-m
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
tags:
  - text-to-image
  - stable-diffusion
  - lora
  - diffusers
  - ateliê-generativo
  - uniceub
instance_prompt: estilo_brasilia
---

# LoRA — Arquitetura Modernista de Brasília

Fine-tuning LoRA do Stable Diffusion v1.5, especializado no estilo visual da **arquitetura
modernista de Brasília** (curvas de concreto branco, céu do cerrado, monumentos icônicos de
Oscar Niemeyer). Projeto acadêmico da disciplina **Inteligência Artificial Generativa e Modelos
Multimodais** — UniCEUB.

## Como usar

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights("andre-lla/lora-brasilia-rank8")

imagem = pipe("estilo_brasilia, um teatro modernista ao entardecer").images[0]
imagem.save("saida.png")
```

**Token de estilo:** sempre inicie o prompt com `estilo_brasilia,` para ativar o estilo aprendido.

## Dataset

- 20-40 fotografias de monumentos públicos de Brasília (Congresso Nacional, Catedral Metropolitana,
  Palácio do Planalto, Museu Nacional, entre outros), coletadas no Wikimedia Commons.
- Todas as imagens possuem licença aberta (CC0 / CC-BY-SA), com proveniência documentada
  (`dados/fontes.csv` no repositório do projeto).
- Legendas geradas com BLIP e **revisadas manualmente** por um humano antes do treino.
- Imagens com pessoas identificáveis em primeiro plano foram descartadas na curadoria, por
  política do projeto.

## Hiperparâmetros de treino

Foram testadas duas configurações, comparando `rank` e `learning_rate`:

| Configuração | rank | learning_rate | max_train_steps (planejado) | max_train_steps (efetivo) |
|---|---|---|---|---|
| A | 8 | 1e-4 | 1500 | 1500 (completo) |
| B | 16 | 5e-5 | 1500 | 1000 (sessão interrompida) |

- `resolution`: 512×512
- `train_batch_size`: 1, `gradient_accumulation_steps`: 4
- `mixed_precision`: fp16
- Prompt de validação usado durante o treino (fora do dataset, para testar generalização):
  `"estilo_brasilia, uma escola modernista com colunas curvas de concreto branco"`

**Configuração escolhida para uso:** rank 16, por apresentar resultados visualmente mais nítidos e
com melhor definição de detalhes (reflexos, texturas de concreto) em relação ao rank 8. Essa
preferência é estética/qualitativa — **não resolve o problema de memorização** descrito na seção
de limitações abaixo, que afeta as duas configurações de forma equivalente.

## Limitações e vieses conhecidos

- **Memorização identificada (achado principal):** ao testar prompts deliberadamente fora do dataset
  de treino (ex.: `estilo_brasilia, museu moderno`, `estilo_brasilia, uma escola com colunas curvas`),
  o modelo não generalizou para um edifício genérico — em vez disso, gerou variações visuais muito
  próximas do Congresso Nacional, o monumento mais representado no dataset. Isso indica que o LoRA
  memorizou parcialmente os monumentos específicos vistos no treino, em vez de aprender um vocabulário
  de estilo (curvas, concreto, proporções) independente do objeto retratado. Esse comportamento foi
  confirmado tanto por inspeção visual quanto por avaliação qualitativa de 3 colegas externos ao
  projeto, que descreveram o modelo como "viciado" em poucas composições.
- **Causas técnicas prováveis:** (i) baixa diversidade de sujeito no dataset — a maioria das imagens
  retrata os mesmos 2-3 monumentos; (ii) capacidade do adaptador (rank 16) suficiente para memorizar
  detalhes específicos com um dataset pequeno e pouco diverso; (iii) ausência de técnicas de
  regularização/prior preservation durante o treino; (iv) legendas que frequentemente nomeiam o
  monumento explicitamente, reforçando a associação entre o token de estilo e a identidade do edifício.
- O dataset é dominado por fotos do Congresso Nacional e da Catedral Metropolitana (monumentos mais
  fotografados/disponíveis), o que agrava o ponto acima.
- A comparação entre as duas configurações de treino não é perfeitamente equivalente, já que a
  configuração B (rank 16) treinou 500 passos a menos que a A (rank 8) devido a uma interrupção
  de sessão — isso deve ser considerado ao comparar os resultados.
- O CLIPScore médio obtido (33,40 em escala 0-100) não capturou esse problema de memorização — prompts
  fora do dataset pontuaram de forma comparável aos prompts de monumentos reais, já que a métrica mede
  alinhamento semântico geral (presença de "colunas", "concreto") e não é sensível à identidade
  específica do sujeito retratado. Isso reforça a importância de avaliação humana complementar às
  métricas automáticas.
- Como toda especialização de estilo via LoRA, o modelo herda quaisquer vieses presentes no
  checkpoint base do Stable Diffusion v1.5.
- Este modelo não deve ser usado para gerar conteúdo que retrate pessoas reais identificáveis,
  nem para fins comerciais sem verificação de licenciamento da imagem base (Stable Diffusion v1.5
  usa a licença CreativeML OpenRAIL-M.

## Recomendações para trabalhos futuros

- Ampliar o dataset com construções modernistas genéricas de Brasília além dos monumentos icônicos
  (superquadras residenciais, escolas-parque, prédios comerciais do Plano Piloto).
- Balancear o número de imagens por monumento, evitando que um ou dois edifícios dominem o aprendizado.
- Revisar as legendas para priorizar a descrição de elementos visuais (curvas, concreto, vãos) em vez
  do nome do edifício retratado.
- Testar rank mais baixo (4) combinado com regularização adicional ou técnicas de prior preservation.
- Ampliar a avaliação humana para o mínimo de 5 avaliadores recomendado, com formulário cego estruturado.

## Uso pretendido

Projeto educacional — geração de imagens estilizadas a partir de temas curtos, como parte de um
pipeline multimodal (texto → imagem → áudio) publicado como demonstração pública.

