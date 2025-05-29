import React from 'react'

const Footer = () => {
  return (
    <footer className='bg-white shadow-inner mt-auto'>
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
            <div className="flex flex-col md:flex-row justify-between items-center">
                <div className="mb-4 md:mb-0">
                    <p className="text-sm text-lightText">
                        &copy; {new Date().getFullYear()} Skin Doctor AI. All rights reserved.
                    </p>
                </div>
                <div className="flex space-x-6">
                    <a href="#" className="text-lightText hover:text-primary">Privacy Policy</a>
                    <a href="#" className='text-lightText hover:text-primary'>Terms of Service</a>
                    <a href="#" className='text-lightText hover:text-primary'>Contact Us</a>
                </div>
            </div>
        </div>
    </footer>
  )
}

export default Footer