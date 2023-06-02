#   READ ME

## Run model
| file name | ability | present support ver. |
| --------------------- | ----------------------------- | ------------------------- |  
| main001.py ~ 005.py | value detect & evaluate IOB accuracy, recall and etc... | main005.py |  
| main0101.py ~ main0202.py | get semantic sentence embedding with contrastive learning & evaluate similarity of sentence & estimate word "slot"| main0202.py |

## Learning model
You run main.py at first, you enable
 ```c 
trained_model = train()
``` 
then model start learning and is stored in `trained_model`  

**WARNING**  
You should comment out 
```
trained_model = MakeSematicSentenceEmbedding()
    trained_model.load_state_dict(torch.load('src/trained_model/generate_semantic_sentence_embedding_model0101.pth'))
```
If you don't comment out, `trained_model` may be overwritten learned model with initialized model

## Save model, Load model
### Save model

You enable
```c
torch.save(trained_model.state_dict(),'src/trained_model/generate_semantic_sentence_embedding_model0101.pth')
```
then trained model `generate_semantic_sentence_embedding_model0101.pth` is saved  
### Load model  
You enable 
```
trained_model = MakeSematicSentenceEmbedding()
    trained_model.load_state_dict(torch.load('src/trained_model/generate_semantic_sentence_embedding_model0101.pth'))
```
then load model `generate_semantic_sentence_embedding_model0101.pth`  
**Tips**  
When you load model, you may comment out 
```
trained_model = train()
```
then you can shorten time of learnning model

## Evaluate only [UNK]
If you want evaluate about [UNK], frag = 1 instead of frag = 0 at line202 in main0202.py



