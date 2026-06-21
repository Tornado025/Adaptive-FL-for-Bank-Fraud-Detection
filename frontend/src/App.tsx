import { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Nav from './components/Nav';
import Splash from './components/Splash';
import Home from './pages/Home';
import HowItWorks from './pages/HowItWorks';
import Playground from './pages/Playground';
import About from './pages/About';

export default function App() {
  const [splashDone, setSplashDone] = useState(false);

  return (
    <BrowserRouter>
      {/* Splash — shown on first load, fades out after ~2s */}
      {!splashDone && <Splash onDone={() => setSplashDone(true)} />}

      <div style={{ opacity: splashDone ? 1 : 0, transition: 'opacity 0.4s ease' }}>
        <Nav />
        <Routes>
          <Route path="/"              element={<Home />} />
          <Route path="/how-it-works"  element={<HowItWorks />} />
          <Route path="/playground"    element={<Playground />} />
          <Route path="/about"         element={<About />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
