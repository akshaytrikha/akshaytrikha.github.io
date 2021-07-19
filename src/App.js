import React from 'react';
import './App.css'
import Painting from './images/painting.jpeg'
import ContactIcon from './components/ContactIcon.js'
import SpotifyIcon from './images/icons/Spotify.png'
import GitHubIcon from './images/icons/GitHub.png'
import LinkedInIcon from './images/icons/LinkedIn.png'
import EmailIcon from './images/icons/Email.png'

// TODO: figure out how to make bullet vertical spacing larger
// TODO: Favorite Reads --> Most Influential Media

function App() {
  return (
    <div className="main-container">
      {/* Header */}
      <header className="page-header">
        <img src={Painting} className="headshot unselectable" alt="headshot"/>
      </header>

      {/* Contact */}
      <section className="contact-icon-group unselectable">
        <a href="https://open.spotify.com/artist/7z8S1uCUgYBYX2lTWx9udV" target="_blank" rel="noopener noreferrer">
          <ContactIcon type={SpotifyIcon} />
        </a>
        <a href="http://github.com/akshaytrikha/" target="_blank" rel="noopener noreferrer">
          <ContactIcon type={GitHubIcon} />
        </a>
        <a href="https://www.linkedin.com/in/akshay-trikha/" target="_blank" rel="noopener noreferrer">
          <ContactIcon type={LinkedInIcon} />
        </a>
        <a href="mailto:atrikha@hmc.edu?">
          <ContactIcon type={EmailIcon} />
        </a>
      </section>

      {/* Persona */}
      <div className="persona">
          {/* Currently */}
          <section className="persona-section">
            <h4 className="unselectable">Currently:</h4>
            <ul>
            <li><span role="img" aria-label="job">💼</span> Looking for a full-time job - <a className="hyperlink" href={process.env.PUBLIC_URL + "pdfs/Akshay_Trikha_Resume.pdf"} target="_blank" rel="noopener noreferrer">Resume</a></li>
              <li><span role="img" aria-label="intern">👨🏾‍💻</span> Working as an intern at <a className="hyperlink" href="https://www.rohankapur.com" target="_blank" rel="noopener noreferrer">Punk House Inc.</a></li>
              <li><span role="img" aria-label="music">🥁</span> Writing music as <a className="hyperlink" href="https://open.spotify.com/artist/7z8S1uCUgYBYX2lTWx9udV" target="_blank" rel="noopener noreferrer">Full Volume Only</a></li>
            </ul>
          </section>

          {/* Recently */}
          <section className="persona-section">
            <h4 className="unselectable">Recently:</h4>
            <ul>
              <li><span role="img" aria-label="research">🔬</span> Did <a className="hyperlink" href={process.env.PUBLIC_URL + "pdfs/Sandia_Clinic_'21_Preprint.pdf"} target="_blank" rel="noopener noreferrer">research</a> on the permittivity of BaTiO<sub>3</sub> nanoparticles (used in capacitors) with <a className="hyperlink" href="https://www.sandia.gov" target="_blank" rel="noopener noreferrer">Sandia National Labs</a></li> 
              <li><span role="img" aria-label="education">📚</span> Got educated at <a className="hyperlink" href="https://www.hmc.edu/about-hmc/" target="_blank" rel="noopener noreferrer">Harvey Mudd College</a></li>
            </ul>
          </section>

          {/* Websites */}
          <section className="persona-section">
            <h4 className="unselectable">Websites that I've made that I'm proud of</h4>
            <ul>
              <li><a className="hyperlink" href="https://styletransfer.art" target="_blank" rel="noopener noreferrer">styletransfer.art</a> - a near real time neural style transfer filter</li>
              <li><a className="hyperlink" href="https://www.cs.hmc.edu/~montanez/projects/futility-of-bias-free-search.html" target="_blank" rel="noopener noreferrer">tinyurl.com/amistad-futility</a> - website to communicate main results from <i>The Futility of Bias-Free Learning and Search</i> in more accessible manner</li>
              <li><a className="hyperlink" href="">this one!</a></li>
            </ul>
          </section>

          {/* Written Work */}
          <section className="persona-section">
            <h4 className="unselectable">Things I've written or helped write that I'm proud of</h4>
            <ul>
              <li><a className="hyperlink" href="https://arxiv.org/pdf/1907.06010.pdf" target="_blank" rel="noopener noreferrer">The Futility of Bias-Free Learning and Search</a></li>
              <li><a className="hyperlink" href={process.env.PUBLIC_URL + "pdfs/Transition_Metal_Oxide_and_Carbide_Pseudocapacitors.pdf"} target="_blank" rel="noopener noreferrer">Transition Metal Oxide and Carbide Pseudocapacitors</a></li>
              <li><a className="hyperlink" href="https://arxiv.org/pdf/1911.04964.pdf" target="_blank" rel="noopener noreferrer">The Bias-Expressivity Tradeoff</a></li>
              <li><a className="hyperlink" href={process.env.PUBLIC_URL + "pdfs/Trump_or_Computer_Dump.pdf"} target="_blank" rel="noopener noreferrer">Trump or Computer Dump?</a></li>
              <li><a className="hyperlink" href="https://open.spotify.com/artist/7z8S1uCUgYBYX2lTWx9udV" target="_blank" rel="noopener noreferrer">Songs for Losers</a></li>
            </ul>
          </section>

          {/* About */}
          <section className="persona-section">
            <h4 className="unselectable">About:</h4>
            <p>I was born in Bombay <span role="img" aria-label="Indian flag">🇮🇳</span>, raised in Hong Kong <span role="img" aria-label="Hong Kong flag">🇭🇰</span>, and grew up in Singapore <span role="img" aria-label="Singaporean flag">🇸🇬</span>. As a result, I can speak Hindi and Chinese too. I'm a fan of people and culture - both creating and preserving it. I've recently immigrated to California <span role="img" aria-label="American flag">🇺🇸</span>, where I'm setting up a new life.</p>
            <p>I enjoy building things (communities, websites, music, solar-arrays, etc.) and often spend my free time on projects to keep learning.</p>
          </section>

          {/* Education */}
          <section className="persona-section">
            <h4 className="unselectable">Education</h4>
            <p>I firmly believe in a liberal eduaction, one that allows you to study ideas seemingly perpendicular to your own field. I majored in CS at Mudd because it gave me the most flexibility to pursue my varied interests.</p>

            <p>Alongside my major, I got to learn about computer engineering, jazz, materials science, philosophy, energy economics, environmental analysis, Asian American history, and modern art.</p>
          </section>

          {/* Readings */}
          <section className="persona-section">
            <h4 className="unselectable">Favorite Reads:</h4>
            <ul>
              <li>The Last Question<a style={{color: "gray"}}> - Isaac Asimov</a></li>
              <li>What I Talk About When I Talk About Running<a style={{color: "gray"}}> - Haruki Murakami</a></li>
              <li>Scar Tissue<a style={{color: "gray"}}> - Anthony Kiedis</a></li>
              <li>The Myth of Sisyphus<a style={{color: "gray"}}> - Albert Camus </a></li>
              <li>When Breath Becomes Air<a style={{color: "gray"}}> - Paul Kalinithi</a></li>
              <li>Being Mortal<a style={{color: "gray"}}> - Atul Gawande</a></li>
              <li>Cats Cradle<a style={{color: "gray"}}> - Kurt Vonnegut</a></li>
              <li>The Subtle Art of Not Giving a F*ck<a style={{color: "gray"}}> - Mark Manson</a></li>
            </ul>
          </section>

          {/* Footer */}
          <footer className="footer unselectable">
            <h1><span role="img" aria-label="rock on">🤘🏾</span></h1>
          </footer>
      </div>
    </div>
  );
}

export default App;
