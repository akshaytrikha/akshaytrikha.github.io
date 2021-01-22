import React from 'react';
import './App.css'
import Painting from './images/painting.jpeg'
import ContactIcon from './components/ContactIcon.js'
import SpotifyIcon from './images/icons/Spotify.png'
import GitHubIcon from './images/icons/GitHub.png'
import LinkedInIcon from './images/icons/LinkedIn.png'
import EmailIcon from './images/icons/Email.png'
// import {} from '@material-ui/core'

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
        <a href="https://open.spotify.com/artist/7z8S1uCUgYBYX2lTWx9udV">
          <ContactIcon type={SpotifyIcon} />
        </a>
        <a href="http://github.com/akshaytrikha/">
          <ContactIcon type={GitHubIcon} />
        </a>
        <a href="https://www.linkedin.com/in/akshay-trikha/">
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
              <li><span role="img" aria-label="music">🥁</span> Writing music as <a className="hyperlink" href="https://open.spotify.com/artist/7z8S1uCUgYBYX2lTWx9udV">Full Volume Only</a></li>
              {/* <li><span role="img" aria-label="research">🔬</span> Doing research at <a className="hyperlink" href="https://arxiv.org/pdf/1907.06010.pdf">AMISTAD lab</a></li> */}
              <li><span role="img" aria-label="research">🔬</span> Doing research on the permittivity of BaTiO<sub>3</sub> nanoparticles (used in capacitors)</li>
              <li><span role="img" aria-label="education">📚</span> Getting educated at <a className="hyperlink" href="https://www.hmc.edu/about-hmc/">Harvey Mudd College</a></li>

            </ul>
          </section>

          {/* Written Work */}
          <section className="persona-section">
            <h4 className="unselectable">Things I've written or helped write that I'm proud of </h4>
            <ul>
              <li><a className="hyperlink" href="https://open.spotify.com/artist/7z8S1uCUgYBYX2lTWx9udV">Songs for Losers</a></li>
              <li><a className="hyperlink" href="https://arxiv.org/pdf/1907.06010.pdf">The Futility of Bias-Free Learning and Search</a></li>
              <li><a className="hyperlink" href="https://drive.google.com/file/d/11RVnSwVW3CGetKaEceJ6C9VOZQL4IZKp/view?usp=sharing">Transition Metal Oxide and Carbide Pseudocapacitors</a></li>
              <li><a className="hyperlink" href="https://arxiv.org/pdf/1911.04964.pdf">The Bias-Expressivity Tradeoff</a></li>
              <li><a className="hyperlink" href="https://drive.google.com/file/d/17myiNEPMZTJs7JgdjB_Q8wqWSuLuAv7m/view?usp=sharing">Trump or Computer Dump?</a></li>
            </ul>
          </section>

          {/* About */}
          <section className="persona-section">
            <h4 className="unselectable">About:</h4>
            <p>I was born in Bombay <span role="img" aria-label="Indian flag">🇮🇳</span>, raised in Hong Kong <span role="img" aria-label="Hong Kong flag">🇭🇰</span>, and grew up in Singapore <span role="img" aria-label="Singaporean flag">🇸🇬</span>. As a result, I can speak Hindi and Chinese too. I'm not a fan of countries, or nationalism, but am a fan of culture - both creating and preserving it. I've recently immigrated to California <span role="img" aria-label="American flag">🇺🇸</span>, where I'm setting up a new life.</p>
            <p>I <i>love</i> building things (communities, music, solar-arrays, etc.). Software was my go-to growing up since it was more accesible than buying hardware, and now I'm trying to utilize the resources at <a className="hyperlink" href="hmc.edu">Harvey Mudd College</a> to build more physical things.</p>
          </section>

          {/* Education */}
          <section className="persona-section">
            <h4 className="unselectable">Education</h4>
            <p>I firmly believe in a liberal eduaction, one that allows you to study ideas seemingly perpendicular to your own field. I'm majoring in CS at Mudd because it gives me the most flexibility to pursue my varied interests.</p>

            <p>Alongside my major, I'm learning about electrical engineering, philosophy, environmental analysis, economics, music, and modern art.</p>
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
