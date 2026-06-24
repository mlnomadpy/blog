# Writing style: making the theory posts read like blogs, not papers

The Arc C posts are technically right and visually rich, but they read like a
results section: claim, evidence, claim. This guide is the fix. It records the
principles (grounded in how the best explanatory writers work), a physics-metaphor
playbook for turning high-dimensional vectors into intuition, a diagnostic
checklist, and a per-post audit of what is currently broken.

Sources that ground this: 3Blue1Brown on starting from motivation and story over
usefulness, Distill on interactive articles for building intuition, and the
energy-landscape literature for the physics-as-intuition bridge (links at the end).

---

## 1. The core diagnosis

A blog post is not a paper. A paper proves; a blog *seduces a reader into
understanding*. The current posts skip the seduction. Three failures recur:

1. **Mechanism before motivation.** They open with the Yat unit, the kernel, the
   denominator, and then assert consequences. The reader is handed the machinery
   before being made to want it. The best explainers do the opposite: a concrete
   puzzle first, the abstraction second, because almost no one wants the general
   thing until a specific thing has made them curious.
2. **No questions, no gaps.** Nothing is ever *asked*. Good explanatory writing
   runs on a curiosity gap: pose a question the reader now needs answered, then
   close it, then open the next. The posts state answers with no question in front
   of them, so there is no tension pulling the reader down the page.
3. **Cold transitions.** Each section restarts from a fact instead of inheriting
   the previous section's ending. There is no single narrator on a single journey,
   just a stack of true paragraphs.

---

## 2. Principles for engaging theory writing

**P1. Motivation before mechanism.** Open every post, and most sections, with a
concrete tension the reader feels in their body, not the abstraction that resolves
it. "To make this network forget a class you delete a few rows" beats "A Yat unit
stores a prototype." Earn the machinery; do not lead with it.

**P2. Run on questions.** Plant a real question and let the section be its answer.
"If the classifier is free, what was training even doing?" pulls harder than "The
accuracy is in the backbone." A section can *open* on the question and *close* on
the answer; the next section opens on the question that answer raises.

**P3. One narrator, one thread.** The end of each section should set up the start
of the next. Transitions are "we just saw X, which is strange, because Y" not a
cold new fact. The reader should never feel a section-break as a reset.

**P4. Intuition first, math as confirmation.** Give the felt picture, let the
reader believe it, then the equation arrives as "and here is why that is exactly
true," not as the source of the belief. Math that lands after conviction reads as
elegant; math that leads reads as homework.

**P5. Name the surprise.** When a result is surprising, build the expectation
first so the reader feels it break. "You would expect the classifier to be the
hard part. It is the free part." Set the trap, then spring it. Wonder over
usefulness: write from delight, not from "this is useful."

**P6. Paragraph openings carry momentum.** The first words of a paragraph are a
handhold. See the smell list below. Open on the live question, a turn ("But"), a
concrete image, or a consequence with stakes, never on the conclusion or an
instruction.

**P7. Show, then label.** Build the intuition in an interactive panel; let the
equation name what the reader just watched happen. The viz is the argument; the
math is the caption.

---

## 3. The physics-metaphor bridge (vectors to intuition)

The reader cannot picture 784 dimensions. That is the actual obstacle in every
one of these posts, and physics is how we get around it. The move is not
decoration; it is the bridge.

**B1. Name the obstacle, then rescue it.** Say it out loud: you cannot see 784
dimensions. Then offer the physical picture as the rescue: but a landscape of
wells, a thing that pulls, a basin you fall into, you *can* see. The metaphor is
introduced as the answer to the reader's "I can't picture this," not as a cute
aside about the formula.

**B2. Make the correspondence exact, and show it as a table.** A loose analogy
loses trust; an exact one earns it. Ours is exact, so say so:

| abstract object | physical object |
| --- | --- |
| prototype $W_u$ | a mass / charge sitting in the space |
| activation $\phi_u(x)$ | the pull you feel from that mass |
| $1/(\lVert x-W_u\rVert^2+\varepsilon)$ | a softened inverse-square law |
| $\varepsilon$ | the N-body softening length |
| class score $s_c = \sum_{u\in c}\phi_u$ | the total field of class $c$'s masses |
| prediction $\arg\max_c s_c$ | falling into the deepest basin |
| decision boundary | the watershed between basins |

**B3. Let the metaphor predict, then confirm with math.** The payoff of a true
metaphor is that it forecasts results before the algebra does. Fields superpose,
so the order you place masses cannot matter (order-independent teaching). Removing
a mass only changes the field near it (exact unlearning). State the physical
prediction first; let the equation confirm it. A metaphor that predicts correctly
is trusted; a metaphor that only re-describes is ignored.

**B4. 2D is the lie that tells the truth.** Show the picture in two dimensions
where the eye works, then say plainly: this is a projection, the real thing lives
in 784 dimensions, but the structure you are looking at is real. Do not pretend
the 2D picture is the space; do use it as the window.

**B5. One physical world across the series.** Keep the same metaphor (masses,
wells, fields, basins, phase changes) consistent post to post so the reader builds
one mental model, not five.

---

## 4. Diagnostic checklist (run before publishing)

**Section-opening test.** Read the first sentence of every section in isolation.
If it could be the first line of a paper's subsection, rewrite it. It should make
a first-time reader either ask a question or feel a tension.

**Paragraph-opening smells (rewrite these):**
- Opens with the conclusion: "Because X, Y." (the reader needed to want Y first)
- Opens with an instruction: "Start with", "Train a", "Take", "Consider", "Note that"
- Opens with a bare subject-noun fact: "The prototypes are", "A standard neuron"
- Opens with a stage direction: "The clean way to see it is", "It is worth", "Look at"
- Opens by announcing: "This is the thesis", "Here is the key point"

**Better openings:** a question ("So what was training doing?"), a turn ("But a
direction is not a thing you can point at"), a concrete image ("Delete twenty rows
of a matrix and a class is gone"), a stake ("Courts now ask models to forget").

**Per-post must-haves:**
- A motivating hook in the first two sentences (a felt problem, not the thesis).
- At least one explicit question the post exists to answer.
- Every section connected to the one before it (no cold restarts).
- The physics bridge introduced as the rescue from high dimensions, not as a note about a formula.
- Math arriving after intuition, never before.

---

## 5. Per-post audit (flagged)

### your-neuron-is-a-picture
- Opens on the thesis, not a puzzle. No "why would anyone want a neuron to be a
  picture?" tension established before the answer is given.
- Section openings are nearly all bare assertions: "This is the entire thesis in
  one image", "The prototypes are points in pixel space", "None of this is a method
  bolted on". State, state, state.
- "Because the output is a kernel-weighted vote over pictures, every prediction
  carries its own explanation" opens the section with its own conclusion.
- The RKHS / representer material is delivered as fact; never framed as "here is
  the thing that should bother you, and here is why it is fine."

### edit-a-network-by-hand
- Best lede of the three (leads with the visceral edit and the two named hard
  problems). Keep that pattern and spread it.
- But the body reverts: "A standard neuron stores a direction w" (fact),
  "Start with a deliberately incomplete model" (methods voice), "Now run it
  backwards" (better, a turn, keep this kind).
- The physics section opens "The denominator repays a second look" — a stage
  direction. It should open on the reader's problem: you cannot picture this space,
  so here is a way you can. The strongest material in the series is introduced as a
  footnote about a fraction.
- Transitions between teach / forget / why-hard are mostly cold restarts.

### train-the-features
- "Train a small convolutional backbone the ordinary way" is pure methods voice,
  no motivation. The reader is told to run an experiment before being told why it
  matters.
- "The clean way to see it is to watch both heads" is a stage direction.
- "Because the head is still a bank of prototypes" opens with the answer.
- The genuinely surprising result (a random network sorts at 74%) is buried mid
  section instead of being set up as a trap and sprung. This is the post's best
  hook and it is thrown away.
- No question drives the post; it is a list of measurements. The title asks a
  question implicitly ("only the features?") but the prose never poses it.

---

## 6. The fix, in order of leverage

1. Rewrite every **section opening** to ask or to turn, never to state (cheapest,
   highest impact).
2. Add a **motivating hook** to the top of your-neuron and train-the-features
   (edit-a-network already has one).
3. Add **connective transitions** so each section inherits the last.
4. Reframe the **physics bridge** as the rescue from high dimensions, with the
   correspondence table, and let it predict before the math confirms.
5. Move **intuition before math** wherever the order is reversed.
6. Promote buried surprises (the random-network 74%) into set-up-and-spring hooks.

---

## Sources

- [3Blue1Brown, About / on explanation](https://www.3blue1brown.com/about/) and [Grant Sanderson on stories and visuals](https://stanforddaily.com/2020/01/24/3blue1brown-creator-grant-sanderson-15-talks-engaging-with-math-using-stories-and-visuals/): start from motivation and intrinsic delight, not the abstraction or usefulness.
- [Distill, Communicating with Interactive Articles](https://distill.pub/2020/communicating-with-interactive-articles/): interactive panels let readers build intuition a static equation cannot.
- [Energy landscapes for machine learning (RSC)](https://pubs.rsc.org/en/content/articlehtml/2017/cp/c7cp01108c) and [Visualising energy landscapes through manifold learning](https://arxiv.org/pdf/2111.07843): the physics-landscape framing and projecting high-dimensional spaces to a viewable picture.
