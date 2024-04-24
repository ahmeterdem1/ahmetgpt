# AhmetGPT

I am copying myself to a language model. Literally.

## Project Overview

So, this is really what it sounds like. I have this little idea in my mind
for quite a while now. I decided to get into action. I will; copy myself,
my social self, into a language model via my WhatsApp messages.

The end goal here is to create a large language model that acts just like
me to an extent that, it passes the Turing Test on imitating me.

This repository contains all my work on this goal. It will take time, but
I will eventually get there. There will be many side projects done on the
way to my full-social-copy, AhmetGPT. 

It is not decided that I will use "GPT". I picked this name because it just
sounds cool.

### Checklist of End Goals

Here are all the things that I expect from AhmetGPT at the end of the project.

- It must pass the Turing Test on imitating me.
- It must be multimodal.
- It must be able to start conversations on its own.
- It must be able to speak English, Turkish and French.
- It must be fast enough to handle live conversations.
- It must be able to understand the concept of time and act accordingly.

## How-to?

How can one copy themselves into ones and zeroes? 

This will of course not be a perfect copy. It will just imitate me. With a 
good engineering of social interaction dynamics and WhatsApp message data,
it should be feasible.

Conversations of people are bidirectional. There are messages incoming, and 
then outgoing. And those messages happen consecutively. Let's say that person
X and I are talking.

Our conversations can be divided into blocks. Incoming-outgoing or outgoing-incoming
blocks are possible. I have named block types as "defense" and "attack".

### Attack Conversation Blocks

Attack conversation blocks start with my messages. After my all transmission of
messages end, person X starts to talk. Attack block continues until the end of
X's continued transmission.

In summary, attack blocks start with my consecutive messages, and ends with the
last consecutive reply that I get.

An edge case which is important is that, my first message(s) that starts the attack
conversation block may belong to a completely different timestamp than the rest of
the conversation. Same goes for the last replies that ends the attack block.

Example:

```

08.35 Me: ...
17.00 Me: ...
17.00 Me: ...
17.01 Me: ...
17.05 Other: ...
17.05 Other: ...
17.06: Other: ...
(Next Day) 10.15 Other: ...

```

Above is a completely valid attack conversation block. We will label this block
with the timestamp of its first message when training the model. This will come
in handy later.

### Defense Conversation Blocks

Defense blocks are the opposite of the attack. They start with the messages of the
person X, and end with my last message after which, again comes person X's message.

Let's directly work on an example:

```

08.35 Other: ...
17.00 Other: ...
17.00 Other: ...
17.01 Other: ...
17.05 Me: ...
17.05 Me: ...
17.06: Me: ...
(Next Day) 10.15 Me: ...

```

This is almost like a received prompt and its answer. However, the last message is
the most important...

### Training

My idea is that, training the model will follow the timeline of my messages. I will
divide all my conversations into blocks and timestamp them with the first message
that the block has.

All conversation blocks will be pooled in a single database. Then when the model
is being trained, blocks will be selected in order of their timestamp. Model will
be trained on the current block that is selected for "n" epochs. Then the model 
will pass to the next block.

Concept of epoch will be applied on batches, instead of the whole dataset. I believe
this will improve the models understanding of timeline. Changes on its weights will
strictly follow the timeline of the events that are happening. Those changes will
mark the models "mind", because model gets trained on the same block for "epoch"
times in a row.

One more thing.

Defense block sometimes ends with a message or messages of mine, that starts a conversation.
When model is trained by defense blocks; it will capture the "prompt", which consists
of the person X's messages. Then it will try to generate a response to that prompt which
will try to be my actual messages. The last message(s) may belong to some later timestamp.

Model will generate all this response in a bulk. But when I code the interactive interface
with this model, I will send you the messages on their timestamps. 

Model will see every messages timestamp during the training. So it must be able to understand
it. When trained with defense blocks, if some of the response that it generates belongs to
tomorrow, I will send it to you tomorrow. The response will halt.

#### Halting Response

I have named this concept, the halting response. 

Different from other language models, my model will generate timestamps for each message
that is in its response that it generates. Some of the response may have a timestamp that
points to a time that is later on. User will see this part of the response when the stamped
time comes. The response will halt.

If before the time comes another prompt is received, the current halted response will be ignored
and will be replaced with the new responses halting part, if it exists.

By collecting all the techniques above, it is possible to train the model so that it understands
conversations, time and it can start conversations without even realizing it!

Model doesn't know if I transmitted its response to you or not. Conversations are bidirectional.
If I can control your next prompt by how I show you the generated response, as it happens in the real life,
I can perfectly simulate a conversation by its all natural and magical depth.

I will train AhmetGPT by mostly defense blocks. But I plan on creating a pure-attack-based separate
model just as an experiment. Attack based strategy doesn't seem to cover prompt-response chain
properly. That is why I find it more interesting than the defense strategy.

## Side projects and more on training

It appears that, considering private chats only; half of all my messages by megabyte, are with just
2 people.

This creates an unbalance of training and test data. This may very well cause that the model not just
imitates me but imitates those 2 people too!

To prevent this, I have collected statistics on all my chats. This contains messages-by-day, ratios
of message counts per person and total length of messages, message density. By this statistics I aim
to weight training and test data to create a balanced environment. However, what I really belive is
that this is unnecessary. 

We are statistical averages of the people that are around us. That is what I believe, at least. If
this is true, then it is perfectly OK to train AhmetGPT on the unbalanced raw data, as it will give
the most accurate representation of the people around me. Models learn the underlying statistical
distribution at the end of the day!

The code to collect statistics on an exported whatsapp chat is given in this repository.

### Message Density

Message density is a concept that again I have invented specifically for this project. It is the average
message/hour value per day, where a day is considered to be 16 hours. 16 hours is picked as the average
time that someone stays awake. You can't message people when you are asleep.

The code for it includes 4 different ways to calculate this parameter. It is calculated via convolution. I
don't consider a days messages only to calculate that specific days message density. Social interactions
are dynamic and have epochs where interaction density fluctuate. These epochs generally cover 2-3 days 
in a bulk. Convolutions aim is to get the weighted average of all this 2-3 day span as the current 
indexed days "message density".

I believe [1, 3, 1] kernel is the best at calculating the incidence-messaging-density. [1, 2, 3, 2, 1]
kernel is also the best at calculating the epoch-messaging-density. It is kind of a choice though.

### Code to Collect Statistics

"wp_statistics.py"

This code outputs a matplotlib plot of given WhatsApp chats statistics. To use it, you need to change a
few things in it. So, knowing a little bit of Python is a prerequisite.

"conversation.txt" should be replaced with your own chat file. You need to also put in usernames as it
appears in the chat file. Those usernames are contact names that you saved people with.

You can delete dictionary keys or add more depending on how many people are in. I have labeled those
fields as "user1", "user2", etc.

Username fields in WhatsApp chat files appear as " - username: ". I have included in this string " - "
and ": " part too since they contribute nothing to the rest of the message. Those are also indicated
in the Python file as is, you can just replace "user1", "user2", etc. parts as in your chat file.

### Finding Closest Matching Message by "Meaning"

"svd_on_messages.py"

This is the coolest side project for now. The application of Singular Value Decomposition
to find the closest matching message by meaning to the given query. 

The used algorithm is "Latent Semantic Indexing". Using the given code, requires a basic
understanding of this algorithm. I am not going to provide specific resources, since
internet literally is a good source.

This file includes a few functions. Except for "find_message()" and "peek_at_index()", all
others need to be called in order if this code is running for the first time for a given
WhatsApp chat. I recommend running them one by one, as they make take super long times
to run.

You just need to fill in the paths for required files. At each function except "find_message()" 
and "peek_at_index()", one of the paths will be the path to a non-existent file that will
be created by the function itself. Follow the document strings given in the code file.

#### Setup

This code requires quite a setup before running. Install all the requirements given in the
"requirements.txt" file. JAX library needs specific conditions to run efficiently that is
different for each operating system. Mostly, follow the instructions given on their official
[website](https://jax.readthedocs.io/en/latest/installation.html). 

For Apple Silicon, you strictly need MacOS 14.4 or above. Also, install jax-metal==0.0.6 with
pip. For "apply_Svd()" function, you need to switch to only CPU. Other than that, you can run 
JAX on GPU. Decomment the specified line in the Python file when switching to CPU only. 

You need to download [Zemberek](https://github.com/ahmetaa/zemberek-nlp) NLP library separately 
as a .jar file. Put this file in the same directory as the Python file. This library includes 
tokenizers and all sorts of things to process Turkish.

Your IDE may give errors after all those steps, trust the process. Only exceptions raised on
runtime are valid.

Running this code may take anywhere from 15 minutes to literal hours depending on your hardware.
Even JAX feels slow at this point.

Feel free to open issues for; issues, ideas or recommendations.


