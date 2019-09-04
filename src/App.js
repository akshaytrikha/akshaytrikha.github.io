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
          {/* About */}
          <section className="persona-section">
            <h4 className="unselectable">About:</h4>
            <p>I was born in Mumbai <span role="img" aria-label="Indian flag">🇮🇳</span>, raised in Hong Kong <span role="img" aria-label="Hong Kong flag">🇭🇰</span>, and grew up in Singapore <span role="img" aria-label="Singaporean flag">🇸🇬</span>. As a result, I can speak Hindi and Chinese too. I'm not a fan of countries, or nationalism, but am a fan of culture - both creating and preserving it. I've recently immigrated to California <span role="img" aria-label="American flag">🇺🇸</span>, where I'm setting up a new life.</p>
            <p>I <i>love</i> building things (music, paintings, solar-arrays, etc.). Software was my go-to growing up since it was more accesible than buying hardware, and now I'm trying to utilize the resources at Harvey Mudd College to build more physical products. I’m majoring in CS at Mudd while concentrating on electrical engineering and economics, while also dabbling with philosophy, environmental analysis, and various artistic disciplines.</p>
          </section>

          {/* Energy */}
          <section className="persona-section">
            <h4 className="unselectable"><span role="img" aria-label="lightning">⚡️:</span></h4>
            <p>In high school I realized the importance of energy in our lives, particularly generating it from sustainable sources. I tried my hand at making my community more sustainable by setting up solar panels that eventually lit up 60 classrooms a day. Now I'm interested in learning more about energy storage and conversion processees.</p>
          </section>

          {/* Currently */}
          <section className="persona-section">
            <h4 className="unselectable">Currently:</h4>
            <ul>
              <li><span role="img" aria-label="sun">☀️</span> Making a portable solar energy cart</li>
              <li><span role="img" aria-label="music">🥁</span> Writing music as <a className="hyperlink" href="https://open.spotify.com/artist/7z8S1uCUgYBYX2lTWx9udV">Full Volume Only</a></li>
              <li><span role="img" aria-label="research">🔬</span> Doing research at <a className="hyperlink" href="https://arxiv.org/pdf/1907.06010.pdf">AMISTAD lab</a></li>
              <li><span role="img" aria-label="education">📚</span> Getting educated at <a className="hyperlink" href="https://www.hmc.edu/about-hmc/">Harvey Mudd College</a></li>
            </ul>
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
