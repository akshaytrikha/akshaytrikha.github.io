import React from 'react';
import '../App.css'

export default class ContactIcon extends React.Component {
  render() {
    return(
      <img src={this.props.type} className="contact-icon" alt={this.props.type.toString()} />
    )
  }
}
