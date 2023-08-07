# GPTnum.py
<p align="center">
  <img src="GPTnum.png" width="500" height="300" alt="Numpy supremus">
</p>

**Heavily inspired by [picoGPT](https://github.com/jaymody/picoGPT) and [llama2.c](https://github.com/karpathy/llama2.c)**

---

* Think GPT is complicated? Guess what? It's not! Introducing GPT-2 with only numpy.
* No keras, no tensorflow, no pytorch, just it's just matrices all the way down baby!!
* All in a single file. No funny build scripts, no sneaky tensorflow imports, just one unreadable python file and the model weights like Kaparthy himself intended
* Banged together over the weekend over a bet made at the office, may god have mercy on you if you try to read the code

---


**Dependencies**
```bash
pip install numpy
```

**Wide range of f̶e̶a̶t̶u̶r̶e̶s̶**

* DOES NOT support any version except the 124M parameter one
* DOES NOT support training
* DOES NOT support GPUs (come on we're running numpy here)
* DOES NOT support batched inference
* DOES NOT support multi-threading (yet)

**Run guide**

* clone the repository from [huggingface](https://huggingface.co/AI-bo/gptnum.py)
```bash
python GPTnum.py
```


**Todo list**

* [ ] Add support for multi-threading because why not
* [ ] Add support for the rest of the GPT-2 model sizes
* [ ] Comment code, make it all readable, with clear typedefs

