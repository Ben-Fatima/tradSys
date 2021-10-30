import React from "react"

import Logo from "../images/logo.svg"
const Menu = () => {
  return (
    <>
      <nav class="sticky top-0 z-10 flex items-center justify-between w-full background">
        <div class="flex flex-shrink-0 mr-6 -space-x-16">
          <img src={Logo} alt="logo" width="60px" className="ml-2"></img>
        </div>
        <div class="w-full block flex-grow lg:flex lg:items-center lg:w-auto">
          <div class="text-sm lg:flex-grow"></div>
          <div className="">
            <button className="menu-item p-3 noto text-white font-semibold block lg:inline-block lg:mt-0 font-normal">
              <i class="fas fa-home p-1 font-normal blue-cl"></i>
            </button>
          </div>
        </div>
      </nav>
    </>
  )
}
export default Menu
