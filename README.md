# neuropong <sup></sup>
This is my trials & errors on training (simulated) neurons to pingpang some pongs. Inspired by this 2022 [paper](https://www.cell.com/neuron/fulltext/S0896-6273(22)00806-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0896627322008066%3Fshowall%3Dtrue) from [Cortical Labs](https://corticallabs.com/).

## ToDos
- [x] make a working game instance in Rust
    - [x] fix edge bug
    - [x] wrap game with PyO3[^1]
- [ ] black vodoo magic with Brian2
    - [x] neuronal test structure
    - [ ] figure out how this shit works
    - [ ] eternal ponging
- [ ] transition from BRIAN to NEURON

## WhyNotDos
- [X] rebound sounds! (bongs are crucial elements)

> [!CAUTION]
> Code reviews might lead to subsequent eye bleaching. The author is yet a novice.

[^1]: Using 0.19.2 because the latest one has some type bound issues with my nighty rust version. 
