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
    link: /assets/pdfs/Akshay_Trikha_Resume.pdf
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

- 👨🏾‍💻 Building and buying LLM-tools at [QuantumScape](https://www.quantumscape.com/){:target="\_blank" :rel="noopener noreferrer"}
- 🔋 (Recently) studied scaling laws for machine learning interatomic potential models at [Berkeley](https://mse.berkeley.edu/){:target="\_blank" :rel="noopener noreferrer"}
- 🛫 Learning how to fly!
<figure>
    <div style="text-align: center;">
        <img src="{{site.url}}/assets/ppl.jpeg" alt="Flying a Cessna 172" width="60%" style="border-radius: 12px;"/>
    </div>
</figure>


<!-- TODO: add nbviewer.org for redox flow battery -->

**Favorite Reads**:

- [The Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf) - Richard Sutton
- The Last Question - Isaac Asimov
- What I Talk About When I Talk About Running - Haruki Murakami
- Scar Tissue - Anthony Kiedis
- The Myth of Sisyphus - Albert Camus
- When Breath Becomes Air - Paul Kalinithi
- Being Mortal - Atul Gawande
- Cats Cradle - Kurt Vonnegut
- The Subtle Art of Not Giving a F\*ck - Mark Manson

🤘🏾
