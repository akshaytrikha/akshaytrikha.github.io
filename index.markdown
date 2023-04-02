---
layout: default
icons:
  - name: github.png
    text: GitHub icon
    link: http://github.com/akshaytrikha/
  - name: linkedin.png
    text: LinkedIn icon
    link: https://www.linkedin.com/in/akshay-trikha/
  - name: resume.png
    text: Resume icon
    link: /assets/pdfs/Resume.pdf
---

<!-- Headshot & Contact -->
<div style="display: flex; flex-direction: column;">
    <div style="text-align: center;">
    <img src="{{ site.baseurl }} /assets/painting.png" alt="Akshay Painting" width="200"/>
        <ul style="margin-right: 3px">
            {% for icon in page.icons %}
                <li style="list-style: none; display: inline; margin-right: 20px;">
                    <a href="{{icon.link}}" target="_blank" rel="noopener noreferrer">
                        <img src="{{ site.baseurl }} /assets/icons/{{ icon.name }}" alt={{icon.text}} width="30"/>
                    </a>
                </li>
            {% endfor %}
        </ul>
    </div>
</div>


**Currently**:
- 👨🏾‍💻 Working as a Machine Learning Engineer at [QuantumScape](https://www.quantumscape.com/){:target="_blank"}
- 🔋 Exploring how to apply deep learning to energy storage

**Recently**:
- 🔬 Did [research](https://doi.org/10.1557/s43580-021-00095-0){:target="_blank"} on the permittivity of BaTiO3 nanoparticles (used in capacitors) with [Sandia National Labs](https://www.sandia.gov){:target="_blank"}
- 📚 Got educated at Harvey Mudd College

<!-- TODO: add nbviewer.org for redox flow battery -->
**Things that I've written that I'm proud of**:
- [Investigating the dielectric properties of barium titanate nanocomposites using transmission electron microscopy image processing](https://doi.org/10.1557/s43580-021-00095-0){:target="_blank"}
- [The Futility of Bias-Free Learning and Search](https://arxiv.org/pdf/1907.06010.pdf){:target="_blank"}
- [Transition Metal Oxide and Carbide Pseudocapacitors](/assets/pdfs/Transition_Metal_Oxide_and_Carbide_Pseudocapacitors.pdf){:target="_blank"}
- [The Bias-Expressivity Tradeoff](https://arxiv.org/pdf/1911.04964.pdf){:target="_blank"}
- [Trump or Computer Dump?](assets/pdfs/Trump_or_Computer_Dump.pdf){:target="_blank"}

**Websites that I've made that I'm proud of**:
- [styletransfer.art](https://styletransfer.art){:target="_blank"} - a near real time neural style transfer filter
- [tinyurl.com/amistad-futility](https://www.cs.hmc.edu/~montanez/projects/futility-of-bias-free-search.html){:target="_blank"} - to communicate main results from The Futility of Bias-Free Learning and Search in more accessible manner

**Favorite Reads**:
- The Last Question - Isaac Asimov
- What I Talk About When I Talk About Running - Haruki Murakami
- Scar Tissue - Anthony Kiedis
- The Myth of Sisyphus - Albert Camus
- When Breath Becomes Air - Paul Kalinithi
- Being Mortal - Atul Gawande
- Cats Cradle - Kurt Vonnegut
- The Subtle Art of Not Giving a F*ck - Mark Manson

🤘🏾