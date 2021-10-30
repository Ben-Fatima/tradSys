import { AnimatePresence } from "framer-motion"
import "./App.css"
import Menu from "./components/Menu.js"
import Formulaire from "./pages/Formulaire"

function App() {
  return (
    <div>
      <AnimatePresence exitBeforeEnter>
        <Menu />
        <Formulaire />
      </AnimatePresence>
    </div>
  )
}

export default App
