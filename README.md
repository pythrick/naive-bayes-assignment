Curso de Especialização de Inteligência Artificial Aplicada

Setor de Educação Profissional e Tecnológica - SEPT

Universidade Federal do Paraná - UFPR

---

**IAA003 - Linguagem de Programação Aplicada**

Prof. Alexander Robert Kutzke

# Exercício de implementação do algoritmo Naive Bayes

Altere o código do arquivo [spam_classifier.py](spam_classifier.py) para adicionar
algumas das seguintes funcionalidades:

- Utilizar a biblioteca NumPy se considerar pertinente;
- Utilizar a biblioteca Pandas se considerar pertinente;
- Analisar o conteúdo da mensagem e não apenas o Assunto;
- Considerar apenas palavras que aparecem um número mínimo de vezes 
  (`min_count`);
- Utilizar apenas radicais das palavras (pesquise por "Porter Stemmer");
- Considerar não apenas presença de palavras, mas outras características:
  - Por exemplo, se a mensagem possuí números:
    - A função `tokenizer` pode retornar *tokens* especiais para isso (por exemplo: 
      `contains:number`).

**Comente seu código indicando as alterações realizadas.**

Você pode, ainda, realizar testes de desempenho para cada uma das alterações realizadas (se for pertinente).

___

## Resolução
- As linhas do script `spam_classifier.py` que tiveram alterações estão comentados com o prefixo `# Change: `.
- Foi utilizado a biblioteca `dynaconf` para fazer o controle das configurações que foram introduzidas no código.
As configurações estão centralizadas no arquivo `settings.toml`, onde é possível habilitar ou desabilitá-las.
- Foi implementado um algoritmo, `run_all_possibilities.py` ,para testar todas as configurações possíveis, de modo a encontrar a configuração 
que tivesse o melhor desempenho. A melhor configuração encontrada está no arquivo `settings.toml`.
