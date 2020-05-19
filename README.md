# Nicer error messages for Python

Other languages such as C++ have gotten quite useful error messages and
diagnostics with tips on how the fix the problems but Python is still stuck
with the bare stacktraces that are very intimidating and often not very helpful.

![Niceback](https://raw.githubusercontent.com/Tronic/niceback/master/docs/with-niceback.png)
**Niceback backtrace shows where the user has terminated the program.**

Niceback hides the irrelevant IPython/notebook internals and concisely shows
what happened (the program was interrupted) and where that happened. This could
further be improved by converting the KeyboardInterrupt message into something
more suitable, like "You stopped the program", but what you see above is just
the default handling that never considered this particular error.

Although IPython and Google Colab developers have done their tweaks to improve
backtraces, it is all too apparent that much remains to be done:

![Colab](https://raw.githubusercontent.com/Tronic/niceback/master/docs/without-niceback.png)
**Standard backtrace from Google Colab.**

Even for the experienced programmer, it is tedious to read through the wall of
text to find the relevant details of what went wrong.

In more complex situations where one might get many screenfuls of standard
traceback, Niceback produces scrollable outputs that concentrate on the relevant
details but also provide variable inspectors on each frame where it may be
relevant:

![Nested exceptions](https://raw.githubusercontent.com/Tronic/niceback/master/docs/nested.png)
**Niceback output with nested exceptions.**


## Usage

At the beginning of your Notebook:

```ipython
!pip install niceback
%load_ext niceback
```

## Background

This project is a proof of concept, showing a modern way to format error
messages in a human-readable manner. Heuristics are used to hide (by default)
irrelevant stack frames and show the actual location of the problem. Since it
would otherwise be impossible to find out the variable contents after the program
has crashed, a variable inspector built into each stack frame quickly reveals
problems with the variables mentioned at the source of error. Care is taken to
add revelant details such as notebook input field numbers and class names not
normally present in Python tracebacks, while hiding overly long paths and other
clutter.

All output is in HTML and as such only works in Jupyter notebooks and other
browser-based systems (this should be useful for web development frameworks as
well). This allows interactivity and much better layout than that of the text
console.

## License

Public Domain or equivalent.

## Help wanted

I won't be able to maintain this all by myself. If you like the idea of nicer
tracebacks, please offer your help in development! Pull requests are welcome
but it would be even better if you could pick up the whole project as your own.

As of now, this project is in no way properly polished for release. Yet, it is
useful enough to such a degree that I always use it in my notebooks, and it
really makes Python development a much smoother experience.
